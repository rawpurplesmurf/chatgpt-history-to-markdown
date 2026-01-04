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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        self.stats = {
            'total': 0,
            'converted': 0,
            'failed': 0,
            'images_copied': 0,
            'attachments_copied': 0
        }
    
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
        if not MERMAID_AVAILABLE:
            return False
        
        try:
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
            
            browser = await launch(headless=True, args=['--no-sandbox'])
            page = await browser.newPage()
            await page.setContent(html_content)
            
            # Wait for mermaid to render
            await page.waitForSelector('svg', {{'timeout': 10000}})
            await asyncio.sleep(1)  # Extra time for rendering
            
            # Get the SVG element and screenshot it
            element = await page.querySelector('.mermaid')
            if element:
                await element.screenshot({{'path': str(output_path), 'omitBackground': True}})
                await browser.close()
                return True
            
            await browser.close()
            return False
            
        except Exception as e:
            logger.warning(f"Failed to render mermaid diagram: {e}")
            return False
    
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
            try:
                dest_file = self.attachments_dir / source_file.name
                shutil.copy2(source_file, dest_file)
                
                # Determine if it's an image
                ext = source_file.suffix.lower()
                if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']:
                    self.stats['images_copied'] += 1
                else:
                    self.stats['attachments_copied'] += 1
                
                return f"attachments/{source_file.name}"
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
            if mermaid_diagrams and MERMAID_AVAILABLE:
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
                        if asyncio.run(self.render_mermaid_to_image(diagram, image_path)):
                            # Replace the code block with an image reference
                            old_block = f"```mermaid\n{diagram}```"
                            new_block = f"![Mermaid Diagram](attachments/{image_name})\n\n<details>\n<summary>View Mermaid Source</summary>\n\n```mermaid\n{diagram}```\n</details>"
                            modified_content = modified_content.replace(old_block, new_block)
                            self.stats['images_copied'] += 1
                            logger.info(f"  ✓ Rendered mermaid diagram to {image_name}")
                    except Exception as e:
                        logger.warning(f"  ✗ Failed to render mermaid diagram: {e}")
                
                message += f"{modified_content}\n\n"
            elif mermaid_diagrams and not MERMAID_AVAILABLE:
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
    
    def convert_conversation(self, conv: Dict, index: int) -> Optional[str]:
        """Convert a single conversation to Markdown."""
        try:
            title = conv.get('title') or "Untitled Conversation"
            create_time = conv.get('create_time')
            update_time = conv.get('update_time')
            
            logger.info(f"[{index}/{self.stats['total']}] Converting: {title}")
            
            # Format timestamps
            date_str = datetime.fromtimestamp(create_time).strftime('%Y-%m-%d') if create_time else "Unknown"
            created = datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S') if create_time else "Unknown"
            updated = datetime.fromtimestamp(update_time).strftime('%Y-%m-%d %H:%M:%S') if update_time else "Unknown"
            
            # Build filename
            clean_title = self.sanitize_filename(title)
            filename = f"{date_str}_{clean_title}.md"
            
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
            output_path = self.output_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            logger.info(f"  ✓ Created {filename} ({messages_processed} messages)")
            self.stats['converted'] += 1
            
            return filename
            
        except Exception as e:
            logger.error(f"  ✗ Failed to convert conversation: {e}", exc_info=True)
            self.stats['failed'] += 1
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
        
        logger.info(f"Loading conversations from: {conversations_file}")
        
        try:
            with open(conversations_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
            
            self.stats['total'] = len(conversations)
            logger.info(f"Found {self.stats['total']} conversations")
            
        except Exception as e:
            logger.error(f"Failed to load conversations.json: {e}")
            sys.exit(1)
        
        # Setup output
        self.setup_directories()
        
        # Convert each conversation
        logger.info("Starting conversion...")
        logger.info("-" * 60)
        
        filenames = []
        
        for i, conv in enumerate(conversations, 1):
            filename = self.convert_conversation(conv, i)
            if filename:
                title = conv.get('title', 'Untitled')
                timestamp = datetime.fromtimestamp(conv.get('create_time', 0))
                filenames.append((title, filename, timestamp))
        
        # Create index
        logger.info("-" * 60)
        if filenames:
            self.create_index(filenames)
        
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
