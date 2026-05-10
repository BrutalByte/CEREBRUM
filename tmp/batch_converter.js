
const { mdToPdf } = require('md-to-pdf');
const fs = require('fs');

const targets = [{"md": "./docs/PDF/__temp_PAPER_001_DSCF_TSC.md", "pdf": "./docs/PDF\\PAPER_001_DSCF_TSC.pdf"}, {"md": "./docs/PDF/__temp_PAPER_002_CSA.md", "pdf": "./docs/PDF\\PAPER_002_CSA.pdf"}, {"md": "./docs/PDF/__temp_PAPER_003_BRIDGE_TWINS.md", "pdf": "./docs/PDF\\PAPER_003_BRIDGE_TWINS.pdf"}, {"md": "./docs/PDF/__temp_PAPER_004_STDP_CAUSAL.md", "pdf": "./docs/PDF\\PAPER_004_STDP_CAUSAL.pdf"}, {"md": "./docs/PDF/__temp_PAPER_005_HOLOGRAPHIC_INDEXING.md", "pdf": "./docs/PDF\\PAPER_005_HOLOGRAPHIC_INDEXING.pdf"}, {"md": "./docs/PDF/__temp_PAPER_006_BAYESIAN_BEAM.md", "pdf": "./docs/PDF\\PAPER_006_BAYESIAN_BEAM.pdf"}, {"md": "./docs/PDF/__temp_PAPER_007_REM_CYCLE.md", "pdf": "./docs/PDF\\PAPER_007_REM_CYCLE.pdf"}, {"md": "./docs/PDF/__temp_PAPER_008_SIGNAL_ENCODER.md", "pdf": "./docs/PDF\\PAPER_008_SIGNAL_ENCODER.pdf"}, {"md": "./docs/PDF/__temp_PAPER_009_THALAMUS.md", "pdf": "./docs/PDF\\PAPER_009_THALAMUS.pdf"}, {"md": "./docs/PDF/__temp_PAPER_010_INFERENCE_VALIDATION.md", "pdf": "./docs/PDF\\PAPER_010_INFERENCE_VALIDATION.pdf"}, {"md": "./docs/PDF/__temp_PAPER_011_CONTRADICTION.md", "pdf": "./docs/PDF\\PAPER_011_CONTRADICTION.pdf"}, {"md": "./docs/PDF/__temp_PAPER_012_REASONING_STUDIO.md", "pdf": "./docs/PDF\\PAPER_012_REASONING_STUDIO.pdf"}, {"md": "./docs/PDF/__temp_PAPER_013_STREAMING_ENGINE.md", "pdf": "./docs/PDF\\PAPER_013_STREAMING_ENGINE.pdf"}, {"md": "./docs/PDF/__temp_PAPER_014_INSIGHT_ENGINE.md", "pdf": "./docs/PDF\\PAPER_014_INSIGHT_ENGINE.pdf"}, {"md": "./docs/PDF/__temp_PAPER_015_ALGORITHMIC_DEPTH.md", "pdf": "./docs/PDF\\PAPER_015_ALGORITHMIC_DEPTH.pdf"}, {"md": "./docs/PDF/__temp_PAPER_016_PRODUCTION_HARDENING.md", "pdf": "./docs/PDF\\PAPER_016_PRODUCTION_HARDENING.pdf"}, {"md": "./docs/PDF/__temp_PAPER_017_CONCLUSION.md", "pdf": "./docs/PDF\\PAPER_017_CONCLUSION.pdf"}, {"md": "./docs/PDF/__temp_SOURCES.md", "pdf": "./docs/PDF\\SOURCES.pdf"}, {"md": "./docs/PDF/__temp_API_REFERENCE.md", "pdf": "./docs/PDF\\API_REFERENCE.pdf"}, {"md": "./docs/PDF/__temp_ARXIV_SUBMISSION_GUIDE.md", "pdf": "./docs/PDF\\ARXIV_SUBMISSION_GUIDE.pdf"}, {"md": "./docs/PDF/__temp_DEPLOYMENT.md", "pdf": "./docs/PDF\\DEPLOYMENT.pdf"}, {"md": "./docs/PDF/__temp_GLOSSARY.md", "pdf": "./docs/PDF\\GLOSSARY.pdf"}, {"md": "./docs/PDF/__temp_INTEGRATION_GUIDE.md", "pdf": "./docs/PDF\\INTEGRATION_GUIDE.pdf"}, {"md": "./docs/PDF/__temp_MIGRATION_GUIDE.md", "pdf": "./docs/PDF\\MIGRATION_GUIDE.pdf"}, {"md": "./docs/PDF/__temp_NOVEL_CONTRIBUTIONS.md", "pdf": "./docs/PDF\\NOVEL_CONTRIBUTIONS.pdf"}, {"md": "./docs/PDF/__temp_PAPER.md", "pdf": "./docs/PDF\\PAPER.pdf"}, {"md": "./docs/PDF/__temp_Parallax_White_Paper.md", "pdf": "./docs/PDF\\Parallax_White_Paper.pdf"}, {"md": "./docs/PDF/__temp_PERFORMANCE_TUNING.md", "pdf": "./docs/PDF\\PERFORMANCE_TUNING.pdf"}, {"md": "./docs/PDF/__temp_REASONING_STUDIO_GUIDE.md", "pdf": "./docs/PDF\\REASONING_STUDIO_GUIDE.pdf"}];
const cssPath = './docs/assets/premium_guide.css';

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
