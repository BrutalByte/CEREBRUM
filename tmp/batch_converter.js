
const { mdToPdf } = require('md-to-pdf');
const fs = require('fs');

const targets = [{"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_001_DSCF_TSC.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_001_DSCF_TSC.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_002_CSA.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_002_CSA.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_003_BRIDGE_TWINS.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_003_BRIDGE_TWINS.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_004_STDP_CAUSAL.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_004_STDP_CAUSAL.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_005_HOLOGRAPHIC_INDEXING.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_005_HOLOGRAPHIC_INDEXING.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_006_BAYESIAN_BEAM.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_006_BAYESIAN_BEAM.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_007_REM_CYCLE.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_007_REM_CYCLE.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_008_SIGNAL_ENCODER.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_008_SIGNAL_ENCODER.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_009_THALAMUS.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_009_THALAMUS.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_010_INFERENCE_VALIDATION.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_010_INFERENCE_VALIDATION.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_011_CONTRADICTION.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_011_CONTRADICTION.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_012_REASONING_STUDIO.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_012_REASONING_STUDIO.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_013_STREAMING_ENGINE.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_013_STREAMING_ENGINE.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_014_INSIGHT_ENGINE.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_014_INSIGHT_ENGINE.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_015_ALGORITHMIC_DEPTH.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_015_ALGORITHMIC_DEPTH.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_016_PRODUCTION_HARDENING.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_016_PRODUCTION_HARDENING.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER_017_CONCLUSION.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER_017_CONCLUSION.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_SOURCES.md", "pdf": "e:/Development/Parallax/docs/PDF\\SOURCES.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_API_REFERENCE.md", "pdf": "e:/Development/Parallax/docs/PDF\\API_REFERENCE.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_ARXIV_SUBMISSION_GUIDE.md", "pdf": "e:/Development/Parallax/docs/PDF\\ARXIV_SUBMISSION_GUIDE.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_DEPLOYMENT.md", "pdf": "e:/Development/Parallax/docs/PDF\\DEPLOYMENT.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_GLOSSARY.md", "pdf": "e:/Development/Parallax/docs/PDF\\GLOSSARY.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_INTEGRATION_GUIDE.md", "pdf": "e:/Development/Parallax/docs/PDF\\INTEGRATION_GUIDE.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_MIGRATION_GUIDE.md", "pdf": "e:/Development/Parallax/docs/PDF\\MIGRATION_GUIDE.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_NOVEL_CONTRIBUTIONS.md", "pdf": "e:/Development/Parallax/docs/PDF\\NOVEL_CONTRIBUTIONS.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PAPER.md", "pdf": "e:/Development/Parallax/docs/PDF\\PAPER.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_Parallax_White_Paper.md", "pdf": "e:/Development/Parallax/docs/PDF\\Parallax_White_Paper.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_PERFORMANCE_TUNING.md", "pdf": "e:/Development/Parallax/docs/PDF\\PERFORMANCE_TUNING.pdf"}, {"md": "e:/Development/Parallax/docs/PDF/__temp_REASONING_STUDIO_GUIDE.md", "pdf": "e:/Development/Parallax/docs/PDF\\REASONING_STUDIO_GUIDE.pdf"}];
const cssPath = 'e:/Development/Parallax/docs/assets/premium_guide.css';

async function convertAll() {
    for (const target of targets) {
        console.log(`Converting ${target.md} -> ${target.pdf}...`);
        try {
            const pdf = await mdToPdf({ path: target.md }, {
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
                fs.writeFileSync(target.pdf, pdf.content);
                // fs.unlinkSync(target.md);
                console.log("   Done.");
            }
        } catch (err) {
            console.error(`   Failed: ${err}`);
        }
    }
}

convertAll();
