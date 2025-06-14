
## README.md
```markdown
# RDKit GPU-Accelerated Featurization CLI

Process multiple SDF files and generate per-file CSVs of molecular descriptors and functional-group counts.

## Features
- **Parallel CPU featurization** via RDKit
- **GPU-accelerated** DataFrame assembly & CSV I/O using cuDF
- **Progress tracking** with tqdm
- Easy CLI interface

## Setup

1. **Clone** the repo:
   ```bash
   git clone <repo-url>
   cd project-root
   ```

2. **Create** and **activate** a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\\Scripts\\activate    # Windows
   ```

3. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Place** your 18 `.sdf` files in `data/`

## Usage

```bash
python src/featurizer.py -i data -o output
```

CSVs will be generated in the `output/` directory.

## Customization
- Edit SMARTS patterns in **src/functional_groups.py**
- Adjust CPU/GPU settings as needed

## License
MIT
```