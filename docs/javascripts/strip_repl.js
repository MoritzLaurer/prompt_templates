document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll('.md-clipboard').forEach(button => {
        const originalClick = button.onclick;
        
        button.onclick = function() {
            const codeBlock = this.closest('pre');
            if (codeBlock) {
                const code = codeBlock.querySelector('code');
                if (code) {
                    // Get the text content
                    let text = code.textContent;
                    
                    // Split into lines
                    let lines = text.split('\n');
                    
                    // Keep only lines that start with >>> or ...
                    lines = lines.filter(line => {
                        const trimmed = line.trim();
                        return trimmed.startsWith('>>>') || trimmed.startsWith('...');
                    });
                    
                    // Remove the prompts
                    lines = lines.map(line => {
                        return line.replace(/^[ \t]*(>>>|\.\.\.) /, '');
                    });
                    
                    // Join back into text
                    text = lines.join('\n').trim();
                    
                    // Copy the modified text
                    navigator.clipboard.writeText(text);
                    return false; // Prevent default copy behavior
                }
            }
            return originalClick.call(this);
        };
    });
}); 