import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


CANVAS_READBACK_PATCH = """
<script>
(function () {
    var originalGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, attrs) {
        if (type === '2d') {
            attrs = Object.assign({}, attrs || {}, { willReadFrequently: true });
        }
        return originalGetContext.call(this, type, attrs);
    };
})();
</script>
"""

def generate_3d_molecule_html(smiles: str) -> str:
    """Takes a SMILES string, generates 3D coordinates, and returns Plotly HTML."""
    
    # 1. Generate 3D Geometry
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "<h1>Invalid SMILES</h1>"
        
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    
    conf = mol.GetConformer()
    
    # 2. Extract Data
    xs, ys, zs, atom_colors, hover_texts = [], [], [], [], []
    
    # Standard CPK colors for basic atoms
    color_map = {1: '#FFFFFF', 6: '#808080', 7: '#0000FF', 8: '#FF0000', 9: '#00FF00'}
    
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        atom = mol.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()
        atomic_num = atom.GetAtomicNum()
        
        xs.append(pos.x)
        ys.append(pos.y)
        zs.append(pos.z)
        atom_colors.append(color_map.get(atomic_num, '#FF00FF')) # Default pink for unknown
        hover_texts.append(f"Atom: {symbol} [{i}]")

    # 3. Extract Bonds for Lines
    b_xs, b_ys, b_zs = [], [], []
    for bond in mol.GetBonds():
        start = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        end = conf.GetAtomPosition(bond.GetEndAtomIdx())
        
        b_xs.extend([start.x, end.x, None])
        b_ys.extend([start.y, end.y, None])
        b_zs.extend([start.z, end.z, None])

    # 4. Build the Plotly Figure
    fig = go.Figure()

    # Add Bonds
    fig.add_trace(go.Scatter3d(
        x=b_xs, y=b_ys, z=b_zs,
        mode='lines',
        line=dict(color='#A0A0A0', width=4),
        hoverinfo='none',
        showlegend=True
    ))

    # Add Atoms
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=12, color=atom_colors, line=dict(width=1, color='#000000')),
        text=hover_texts,
        hoverinfo='text',
        showlegend=True
    ))

    # Apply Dark Mode Styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="#1e1e1e"
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(26,28,43,0.75)',
            bordercolor='#3A3F49',
            borderwidth=1,
            font=dict(color="#F2F4F8", size=11)
        )
    )

    html = fig.to_html(include_plotlyjs='cdn', full_html=True)
    if "<head>" in html:
        html = html.replace("<head>", "<head>" + CANVAS_READBACK_PATCH, 1)
    return html