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

def _bond_key(i: int, j: int):
    return (i, j) if i <= j else (j, i)


def generate_3d_molecule_html(
    smiles: str,
    atom_contributions=None,
    attention_bonds=None,
    attention_mode: bool = False,
) -> str:
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
    atom_contributions = atom_contributions or []
    
    # Standard CPK colors for basic atoms
    color_map = {1: '#FAF9F6', 6: '#8A8178', 7: '#3A68B7', 8: '#D94F2A', 9: '#6BBF59'}
    
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        atom = mol.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()
        atomic_num = atom.GetAtomicNum()
        
        xs.append(pos.x)
        ys.append(pos.y)
        zs.append(pos.z)
        atom_colors.append(color_map.get(atomic_num, '#CC5500'))
        if i < len(atom_contributions):
            contribution = float(atom_contributions[i])
            hover_texts.append(
                f"Atom: {symbol} [{i}]<br>Contribution: {contribution:.3f}"
            )
        else:
            hover_texts.append(f"Atom: {symbol} [{i}]<br>Contribution: n/a")

    # 3. Extract Bonds for Lines
    base_b_xs, base_b_ys, base_b_zs = [], [], []
    focus_b_xs, focus_b_ys, focus_b_zs = [], [], []
    attention_lookup = {}
    for b in (attention_bonds or []):
        i = int(b.get('begin', -1))
        j = int(b.get('end', -1))
        score = float(b.get('score', 0.0))
        if i >= 0 and j >= 0:
            attention_lookup[_bond_key(i, j)] = score

    if attention_lookup:
        scores = np.asarray(list(attention_lookup.values()), dtype=np.float32)
        threshold = float(np.quantile(scores, 0.75))
    else:
        threshold = 1.1

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        start = conf.GetAtomPosition(i)
        end = conf.GetAtomPosition(j)

        score = attention_lookup.get(_bond_key(i, j), 0.0)
        is_focus = attention_mode and score >= threshold

        if is_focus:
            focus_b_xs.extend([start.x, end.x, None])
            focus_b_ys.extend([start.y, end.y, None])
            focus_b_zs.extend([start.z, end.z, None])
        else:
            base_b_xs.extend([start.x, end.x, None])
            base_b_ys.extend([start.y, end.y, None])
            base_b_zs.extend([start.z, end.z, None])

    # 4. Build the Plotly Figure
    fig = go.Figure()

    # Add Bonds (normal or faded in attention mode)
    fig.add_trace(go.Scatter3d(
        x=base_b_xs, y=base_b_ys, z=base_b_zs,
        mode='lines',
        line=dict(color='rgba(122, 90, 66, 0.20)' if attention_mode else '#BFAFA0', width=4),
        hoverinfo='none',
        name='Bonds',
        showlegend=True
    ))

    if attention_mode:
        fig.add_trace(go.Scatter3d(
            x=focus_b_xs, y=focus_b_ys, z=focus_b_zs,
            mode='lines',
            line=dict(color='#CC5500', width=8),
            hoverinfo='none',
            name='Attention Focus',
            showlegend=True
        ))

    # Add Atoms
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=12, color=atom_colors, line=dict(width=1, color='#6B5A4D')),
        text=hover_texts,
        hoverinfo='text',
        name='Atoms',
        showlegend=True
    ))

    # Apply Dark Mode Styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#FAF9F6",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="#FAF9F6"
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,253,248,0.88)',
            bordercolor='#E0C7B1',
            borderwidth=1,
            font=dict(color="#4A3A2A", size=11)
        )
    )

    html = fig.to_html(include_plotlyjs='cdn', full_html=True)
    if "<head>" in html:
        html = html.replace("<head>", "<head>" + CANVAS_READBACK_PATCH, 1)
    return html
