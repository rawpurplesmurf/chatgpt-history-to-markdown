#!/usr/bin/env python3
"""
ChatGPT History Converter
Converts ChatGPT conversation exports to organized Markdown documentation.
"""

import json
import os
import re
import shutil
import sys
import logging
import asyncio
import hashlib
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    from pyppeteer import launch
    MERMAID_AVAILABLE = True
except ImportError:
    MERMAID_AVAILABLE = False
    logging.warning("pyppeteer not installed - mermaid diagrams will not be rendered. Install with: pip install pyppeteer")


class ChatGPTConverter:
    """Converts ChatGPT conversation history to Markdown files."""
    
    def __init__(self, source_dir: str = 'chatgpt-history-source', output_dir: str = 'markdown'):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.attachments_dir = self.output_dir / 'attachments'
        self.mermaid_enabled = MERMAID_AVAILABLE and self._env_truthy('CHATGPT_HISTORY_RENDER_MERMAID', default=True)
        self.build_search_index = self._env_truthy('CHATGPT_HISTORY_BUILD_SEARCH_INDEX', default=True)
        self._stats_lock = threading.Lock()
        self._attachment_locks_guard = threading.Lock()
        self._attachment_locks: Dict[str, threading.Lock] = {}
        self._mermaid_lock = threading.Lock()
        self.stats = {
            'total': 0,
            'converted': 0,
            'failed': 0,
            'images_copied': 0,
            'attachments_copied': 0
        }

    def _env_truthy(self, name: str, default: bool = True) -> bool:
        """Read a boolean-like environment variable."""
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() not in ('0', 'false', 'no', 'off', '')

    def _increment_stat(self, key: str, amount: int = 1) -> None:
        with self._stats_lock:
            self.stats[key] += amount

    def _get_attachment_lock(self, name: str) -> threading.Lock:
        with self._attachment_locks_guard:
            lock = self._attachment_locks.get(name)
            if lock is None:
                lock = threading.Lock()
                self._attachment_locks[name] = lock
            return lock

    def _markdown_matches_conversation_id(self, path: Path, conv_id: str) -> bool:
        """Best-effort check to keep reruns idempotent (overwrite same conversation output)."""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                head = f.read(4096)
            return f"**Conversation ID:** `{conv_id}`" in head
        except Exception:
            return False

    def _iter_conversations_streaming(self, conversations_file: Path, chunk_size: int = 1024 * 1024) -> Iterable[Dict]:
        """
        Stream items from a top-level JSON array (ChatGPT conversations.json) without loading it all into memory.

        This assumes conversations.json is a JSON array of objects.
        """
        decoder = json.JSONDecoder()
        buffer = ""

        with open(conversations_file, "r", encoding="utf-8") as f:
            # Read until we have non-whitespace to find the opening '['
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buffer += chunk
                buffer = buffer.lstrip()
                if buffer:
                    break

            if not buffer:
                return

            if buffer[0] != "[":
                raise ValueError("Expected conversations.json to be a JSON array")

            buffer = buffer[1:]

            while True:
                buffer = buffer.lstrip()

                if not buffer:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise ValueError("Unexpected end of file while reading conversations.json")
                    buffer += chunk
                    continue

                if buffer[0] == "]":
                    return

                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue

                while True:
                    try:
                        item, end = decoder.raw_decode(buffer)
                        yield item
                        buffer = buffer[end:]
                        break
                    except json.JSONDecodeError:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            raise
                        buffer += chunk

    def _count_conversations_streaming(self, conversations_file: Path) -> int:
        count = 0
        for _ in self._iter_conversations_streaming(conversations_file):
            count += 1
        return count
    
    def setup_directories(self) -> None:
        """Create output directories if they don't exist."""
        logger.info(f"Setting up output directory: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.attachments_dir.mkdir(parents=True, exist_ok=True)
    
    def sanitize_filename(self, title: str) -> str:
        """Clean title for use as filename."""
        # Remove or replace problematic characters
        clean = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_')]).rstrip()
        return clean.replace(' ', '_') if clean else "Untitled"
    
    def detect_mermaid_diagrams(self, text: str) -> List[str]:
        """Extract mermaid diagram code blocks from text."""
        pattern = r'```mermaid\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches
    
    async def render_mermaid_to_image(self, mermaid_code: str, output_path: Path) -> bool:
        """Render a mermaid diagram to an image file using pyppeteer."""
        if not self.mermaid_enabled:
            return False
        
        # Create HTML with mermaid
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
            </script>
        </head>
        <body>
            <div class="mermaid">
{mermaid_code}
            </div>
        </body>
        </html>
        """

        browser = None
        try:
            launch_kwargs = {'headless': True, 'args': ['--no-sandbox']}
            chrome_path = os.environ.get('CHATGPT_HISTORY_CHROME_PATH')
            if chrome_path:
                launch_kwargs['executablePath'] = chrome_path

            browser = await launch(**launch_kwargs)
            page = await browser.newPage()
            await page.setContent(html_content)

            # Wait for mermaid to render
            await page.waitForSelector('svg', {'timeout': 10000})
            await asyncio.sleep(1)  # Extra time for rendering

            # Get the SVG element and screenshot it
            element = await page.querySelector('.mermaid')
            if not element:
                return False

            await element.screenshot({'path': str(output_path), 'omitBackground': True})
            return True
        finally:
            if browser:
                try:
                    await browser.close()
                except Exception:
                    pass
    
    def copy_attachment(self, file_id: str) -> Optional[str]:
        """Copy attachment file from source to output directory."""
        # Look for the file in source directory
        source_file = self.source_dir / file_id
        
        if not source_file.exists():
            # Try common patterns
            for pattern in [f'file-{file_id}*', f'*{file_id}*']:
                matches = list(self.source_dir.glob(pattern))
                if matches:
                    source_file = matches[0]
                    break
        
        if source_file.exists():
            dest_name = source_file.name
            dest_file = self.attachments_dir / dest_name
            dest_rel = f"attachments/{dest_name}"
            lock = self._get_attachment_lock(dest_name)
            try:
                with lock:
                    if dest_file.exists():
                        return dest_rel

                    shutil.copy2(source_file, dest_file)

                    # Determine if it's an image
                    ext = source_file.suffix.lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']:
                        self._increment_stat('images_copied')
                    else:
                        self._increment_stat('attachments_copied')

                    return dest_rel
            except Exception as e:
                logger.warning(f"Failed to copy attachment {file_id}: {e}")
        
        return None
    
    def process_message_parts(self, parts: List) -> Tuple[str, List[str]]:
        """Process message parts, extracting text and file references."""
        text_content = []
        attachments = []
        
        for part in parts:
            if isinstance(part, str):
                text_content.append(part)
            elif isinstance(part, dict):
                # Handle image or file references
                if 'asset_pointer' in part or 'file_id' in part:
                    file_id = part.get('asset_pointer') or part.get('file_id')
                    if file_id:
                        attachments.append(file_id)
        
        return '\n'.join(text_content), attachments
    
    def format_message(self, role: str, content: str, attachments: List[str]) -> str:
        """Format a message with role, content, and attachments."""
        role_emoji = {
            'user': '👤',
            'assistant': '🤖',
            'system': '⚙️',
            'tool': '🔧'
        }
        
        role_label = f"{role_emoji.get(role, '💬')} **{role.title()}**"
        
        # Build message content
        message = f"### {role_label}\n\n"
        
        if content.strip():
            # Check for mermaid diagrams and render them
            mermaid_diagrams = self.detect_mermaid_diagrams(content)
            if mermaid_diagrams and self.mermaid_enabled:
                logger.info(f"  Found {len(mermaid_diagrams)} mermaid diagram(s) - rendering...")
                
                # Replace mermaid blocks with image references
                modified_content = content
                for i, diagram in enumerate(mermaid_diagrams):
                    # Create unique filename based on diagram hash
                    diagram_hash = hashlib.md5(diagram.encode()).hexdigest()[:8]
                    image_name = f"mermaid_{diagram_hash}.png"
                    image_path = self.attachments_dir / image_name
                    
                    # Render diagram
                    try:
                        with self._mermaid_lock:
                            if not self.mermaid_enabled:
                                break
                            already_exists = image_path.exists()
                            if already_exists:
                                rendered = True
                            else:
                                rendered = asyncio.run(self.render_mermaid_to_image(diagram, image_path))

                        if rendered:
                            # Replace the code block with an image reference
                            old_block = f"```mermaid\n{diagram}```"
                            new_block = f"![Mermaid Diagram](attachments/{image_name})\n\n<details>\n<summary>View Mermaid Source</summary>\n\n```mermaid\n{diagram}```\n</details>"
                            modified_content = modified_content.replace(old_block, new_block)
                            if not already_exists:
                                self._increment_stat('images_copied')
                            logger.info(f"  ✓ Rendered mermaid diagram to {image_name}")
                    except Exception as e:
                        error_msg = str(e)
                        launch_failed_signals = (
                            "Browser closed unexpectedly",
                            "Failed to launch the browser process",
                            "Bad CPU type in executable",
                            "Exec format error",
                            "incompatible architecture",
                            "wrong architecture",
                            "not supported on this Mac",
                        )
                        if any(signal in error_msg for signal in launch_failed_signals) or isinstance(e, OSError):
                            self.mermaid_enabled = False
                            logger.warning(
                                "Mermaid rendering disabled (pyppeteer/Chromium failed to start). "
                                "Leaving ```mermaid``` blocks as-is. "
                                "Set CHATGPT_HISTORY_RENDER_MERMAID=0 to silence, or set "
                                "CHATGPT_HISTORY_CHROME_PATH to a working Chrome/Chromium binary."
                            )
                            break
                        logger.warning(f"  ✗ Failed to render mermaid diagram: {e}")
                
                message += f"{modified_content}\n\n"
            elif mermaid_diagrams and not self.mermaid_enabled:
                logger.info(f"  Found {len(mermaid_diagrams)} mermaid diagram(s) but rendering unavailable")
                message += f"{content}\n\n"
            else:
                message += f"{content}\n\n"
        
        # Add attachments
        for attachment in attachments:
            attachment_path = self.copy_attachment(attachment)
            if attachment_path:
                # Check if it's an image
                ext = Path(attachment_path).suffix.lower()
                if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']:
                    message += f"![Attachment]({attachment_path})\n\n"
                else:
                    message += f"📎 [Attachment]({attachment_path})\n\n"
        
        return message
    
    def convert_conversation(self, conv: Dict, index: int, total: Optional[int] = None) -> Optional[str]:
        """Convert a single conversation to Markdown."""
        try:
            title = conv.get('title') or "Untitled Conversation"
            create_time = conv.get('create_time')
            update_time = conv.get('update_time')
            
            if total:
                logger.info(f"[{index}/{total}] Converting: {title}")
            else:
                logger.info(f"[{index}] Converting: {title}")
            
            # Format timestamps
            date_str = datetime.fromtimestamp(create_time).strftime('%Y-%m-%d') if create_time else "Unknown"
            created = datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S') if create_time else "Unknown"
            updated = datetime.fromtimestamp(update_time).strftime('%Y-%m-%d %H:%M:%S') if update_time else "Unknown"
            
            # Build filename
            clean_title = self.sanitize_filename(title)
            base_filename = f"{date_str}_{clean_title}.md"
            
            # Build markdown content
            md_content = f"# {title}\n\n"
            md_content += f"**Created:** {created}  \n"
            md_content += f"**Updated:** {updated}  \n"
            md_content += f"**Conversation ID:** `{conv.get('conversation_id', 'N/A')}`\n\n"
            md_content += "---\n\n"
            
            # Process messages
            mapping = conv.get('mapping', {})
            if not mapping:
                logger.warning(f"  No messages found in conversation")
                return None
            
            # Build message tree and extract in order
            messages_processed = 0
            for node_id in mapping:
                node = mapping[node_id]
                message = node.get('message')
                
                if not message:
                    continue
                
                # Skip system/hidden messages
                if message.get('metadata', {}).get('is_visually_hidden_from_conversation'):
                    continue
                
                author = message.get('author', {})
                role = author.get('role', 'unknown')
                
                # Skip tool calls (unless they have content)
                if role == 'tool' and not message.get('content', {}).get('parts'):
                    continue
                
                content_data = message.get('content', {})
                parts = content_data.get('parts', [])
                
                if not parts:
                    continue
                
                # Process message content
                text, attachments = self.process_message_parts(parts)
                
                if text.strip() or attachments:
                    md_content += self.format_message(role, text, attachments)
                    messages_processed += 1
            
            if messages_processed == 0:
                logger.warning(f"  No displayable messages found")
                return None
            
            # Write the file
            conv_id = conv.get('conversation_id') or conv.get('id')
            conv_id_str = str(conv_id) if conv_id is not None else ""

            # Avoid overwriting when multiple conversations resolve to the same filename, but
            # allow overwriting when rerunning for the same conversation id.
            stem = Path(base_filename).stem
            suffix = Path(base_filename).suffix
            filename: Optional[str] = None

            def try_write(candidate: str) -> Optional[str]:
                output_path = self.output_dir / candidate

                if conv_id is not None and output_path.exists() and self._markdown_matches_conversation_id(output_path, conv_id_str):
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(md_content)
                    return candidate

                try:
                    with open(output_path, 'x', encoding='utf-8') as f:
                        f.write(md_content)
                    return candidate
                except FileExistsError:
                    return None

            filename = try_write(base_filename)
            if filename is None and conv_id is not None:
                filename = try_write(f"{stem}_{conv_id_str[:8]}{suffix}")
            if filename is None:
                for n in range(2, 1000):
                    filename = try_write(f"{stem}_{n}{suffix}")
                    if filename is not None:
                        break

            if filename is None:
                # Last resort: overwrite base filename
                filename = base_filename
                with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
                    f.write(md_content)
            
            logger.info(f"  ✓ Created {filename} ({messages_processed} messages)")
            self._increment_stat('converted')
            
            return filename
            
        except Exception as e:
            logger.error(f"  ✗ Failed to convert conversation: {e}", exc_info=True)
            self._increment_stat('failed')
            return None
    
    def create_index(self, filenames: List[Tuple[str, str, datetime]]) -> None:
        """Create navigation index file."""
        logger.info("Creating index.md")
        
        # Sort by date (newest first)
        filenames.sort(key=lambda x: x[2], reverse=True)
        
        index_content = "# 📚 ChatGPT Conversation Archive\n\n"
        index_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        index_content += f"Total conversations: {len(filenames)}\n\n"
        index_content += "---\n\n"
        
        # Group by year-month
        by_month = {}
        for title, filename, timestamp in filenames:
            month_key = timestamp.strftime('%Y-%m')
            if month_key not in by_month:
                by_month[month_key] = []
            by_month[month_key].append((title, filename, timestamp))
        
        # Write grouped entries
        for month in sorted(by_month.keys(), reverse=True):
            month_name = datetime.strptime(month, '%Y-%m').strftime('%B %Y')
            index_content += f"## {month_name}\n\n"
            
            for title, filename, timestamp in by_month[month]:
                date_str = timestamp.strftime('%Y-%m-%d')
                index_content += f"- [{title}]({filename}) - *{date_str}*\n"
            
            index_content += "\n"
        
        # Write index file
        with open(self.output_dir / 'index.md', 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        logger.info("✓ Index created successfully")
    
    def convert(self) -> None:
        """Main conversion process."""
        logger.info("=" * 60)
        logger.info("ChatGPT History Converter")
        logger.info("=" * 60)
        
        # Load conversations
        conversations_file = self.source_dir / 'conversations.json'
        
        if not conversations_file.exists():
            logger.error(f"Conversations file not found: {conversations_file}")
            sys.exit(1)
        
        workers_env = os.environ.get('CHATGPT_HISTORY_WORKERS', '').strip()
        try:
            workers = int(workers_env) if workers_env else 1
        except ValueError:
            logger.warning(f"Invalid CHATGPT_HISTORY_WORKERS={workers_env!r}; defaulting to 1")
            workers = 1
        workers = max(1, workers)

        stream_json = self._env_truthy('CHATGPT_HISTORY_STREAM', default=False)
        count_total = self._env_truthy('CHATGPT_HISTORY_COUNT_TOTAL', default=False)

        logger.info(f"Loading conversations from: {conversations_file}")

        conversations_iter: Iterable[Dict]
        total: Optional[int] = None

        try:
            if stream_json:
                if count_total:
                    logger.info("Counting conversations (streaming)...")
                    total = self._count_conversations_streaming(conversations_file)
                    self.stats['total'] = total
                    logger.info(f"Found {total} conversations")
                else:
                    logger.info("Streaming conversations (low-memory mode)...")
                conversations_iter = self._iter_conversations_streaming(conversations_file)
            else:
                with open(conversations_file, 'r', encoding='utf-8') as f:
                    conversations = json.load(f)
                total = len(conversations)
                self.stats['total'] = total
                logger.info(f"Found {total} conversations")
                conversations_iter = conversations
        except Exception as e:
            logger.error(f"Failed to load conversations.json: {e}")
            sys.exit(1)
        
        # Setup output
        self.setup_directories()
        
        # Convert each conversation
        logger.info("Starting conversion...")
        logger.info("-" * 60)
        
        filenames: List[Tuple[str, str, datetime]] = []

        def run_one(conversation: Dict, idx: int) -> Optional[Tuple[str, str, datetime]]:
            filename = self.convert_conversation(conversation, idx, total=total)
            if not filename:
                return None
            title = conversation.get('title', 'Untitled')
            timestamp = datetime.fromtimestamp(conversation.get('create_time', 0))
            return (title, filename, timestamp)

        if workers == 1:
            processed = 0
            for i, conv in enumerate(conversations_iter, 1):
                processed += 1
                result = run_one(conv, i)
                if result:
                    filenames.append(result)
            if total is None:
                self.stats['total'] = processed
        else:
            processed = 0
            in_flight = set()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for i, conv in enumerate(conversations_iter, 1):
                    processed += 1
                    in_flight.add(executor.submit(run_one, conv, i))
                    if len(in_flight) >= workers * 2:
                        done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                        for future in done:
                            result = future.result()
                            if result:
                                filenames.append(result)

                while in_flight:
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        result = future.result()
                        if result:
                            filenames.append(result)

            if total is None:
                self.stats['total'] = processed
        
        # Create index
        logger.info("-" * 60)
        if filenames:
            self.create_index(filenames)

        if self.build_search_index:
            try:
                from search_index import SearchIndexBuilder

                SearchIndexBuilder(self.output_dir).build()
            except Exception as e:
                logger.warning(f"Search index build failed: {e}")
        else:
            logger.info("Search index build disabled (CHATGPT_HISTORY_BUILD_SEARCH_INDEX=0)")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Conversion Summary")
        logger.info("=" * 60)
        logger.info(f"Total conversations: {self.stats['total']}")
        logger.info(f"Successfully converted: {self.stats['converted']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Images copied: {self.stats['images_copied']}")
        logger.info(f"Attachments copied: {self.stats['attachments_copied']}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        logger.info("=" * 60)
        
        if self.stats['failed'] > 0:
            logger.warning(f"⚠️  {self.stats['failed']} conversation(s) failed to convert")
        
        if self.stats['converted'] > 0:
            logger.info("✅ Conversion complete!")
        else:
            logger.error("❌ No conversations were converted")
            sys.exit(1)


if __name__ == '__main__':
    converter = ChatGPTConverter()
    converter.convert()
