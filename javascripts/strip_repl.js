// Create a function to handle the copy operation
function handleCopy(button) {
    const codeBlock = button.closest('pre');
    if (!codeBlock) return;

    const code = codeBlock.querySelector('code');
    if (!code) return;

    button.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();

        // Get and process the text
        let text = code.textContent;
        let lines = text.split('\n');
        
        // Remove REPL prompts while keeping the actual code
        lines = lines
            .map(line => line.replace(/^[\t ]*(>>>|\.\.\.) ?/, ''))
            .filter(line => line.length > 0);
        
        // Join lines and copy to clipboard
        const cleanedText = lines.join('\n').trim();
        navigator.clipboard.writeText(cleanedText);
        
        return false;
    }, { capture: true });  // Use capture to ensure our handler runs first
}

// Set up observer to watch for new copy buttons
const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
        if (mutation.type === 'childList') {
            mutation.addedNodes.forEach(node => {
                if (node.nodeType === 1) {  // Element node
                    // Handle newly added copy buttons
                    node.querySelectorAll('.md-clipboard').forEach(handleCopy);
                }
            });
        }
    }
});

// Initial setup
document.addEventListener('DOMContentLoaded', () => {
    // Handle existing copy buttons
    document.querySelectorAll('.md-clipboard').forEach(handleCopy);
    
    // Start observing the document for new copy buttons
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
}); 