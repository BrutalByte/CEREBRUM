const { mdToPdf } = require('md-to-pdf');
const fs = require('fs');
const path = require('path');

const mdPath = 'e:/Development/Parallax/docs/Parallax_Plain_Language_Guide_Professional.md';
const cssPath = 'e:/Development/Parallax/docs/assets/premium_guide.css';
const outputPath = 'e:/Development/Parallax/docs/Parallax_Plain_Language_Guide_Professional.pdf';

(async () => {
    try {
        console.log("Generating Professional PDF via Node engine...");
        const pdf = await mdToPdf({ path: mdPath }, {
            stylesheet: cssPath,
            pdf_options: {
                format: 'A4',
                margin: '0mm',
                printBackground: true
            },
            launch_options: {
                args: ['--no-sandbox', '--disable-setuid-sandbox', '--allow-file-access-from-files']
            }
        });

        if (pdf) {
            fs.writeFileSync(outputPath, pdf.content);
            console.log(`Success! Professional PDF saved at: ${outputPath}`);
        }
    } catch (err) {
        console.error("PDF Generation Failed:", err);
    }
})();
