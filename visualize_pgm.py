"""
Visualization for the Heterogeneous PGM
Creates plots showing the network structure and relationships between variables.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from heterogeneous_pgm import HeterogeneousPGM

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

# Create the model
model = HeterogeneousPGM()

# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 10))

# ============================================================
# 1. Network Structure (top left)
# ============================================================
ax1 = plt.subplot(2, 3, 1)
G = nx.DiGraph()
G.add_edges_from([
    ('Study Hours\n(continuous)', 'Mood\n(discrete)'),
    ('Study Hours\n(continuous)', 'Exam Score\n(continuous)'),
    ('Mood\n(discrete)', 'Exam Score\n(continuous)')
])

pos = {
    'Study Hours\n(continuous)': (0, 1),
    'Mood\n(discrete)': (1, 0.5),
    'Exam Score\n(continuous)': (2, 1)
}

# Color nodes by type
node_colors = ['lightblue', 'lightcoral', 'lightblue']
nx.draw(G, pos, with_labels=True, node_color=node_colors,
        node_size=3000, font_size=9, font_weight='bold',
        arrows=True, arrowsize=20, arrowstyle='->', ax=ax1,
        edge_color='gray', width=2)
ax1.set_title('Heterogeneous PGM Structure', fontsize=12, fontweight='bold')
ax1.text(0.5, -0.3, 'Blue = Continuous, Red = Discrete',
         transform=ax1.transAxes, ha='center', fontsize=9, style='italic')

# ============================================================
# 2. P(Mood = Good | Study Hours) - Sigmoid curve
# ============================================================
ax2 = plt.subplot(2, 3, 2)
study_range = np.linspace(0, 10, 100)
mood_probs = [model.infer_mood_probability(h)['Good'] for h in study_range]

ax2.plot(study_range, mood_probs, 'b-', linewidth=3, label='P(Mood = Good)')
ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='P = 0.5')
ax2.axvline(x=4.0, color='g', linestyle='--', alpha=0.5, label='Threshold (4h)')
ax2.fill_between(study_range, mood_probs, alpha=0.3)
ax2.set_xlabel('Study Hours', fontsize=11, fontweight='bold')
ax2.set_ylabel('Probability', fontsize=11, fontweight='bold')
ax2.set_title('Mood Probability vs Study Hours\n(Discrete given Continuous)',
              fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 1)

# ============================================================
# 3. Expected Exam Score vs Study Hours (for both moods)
# ============================================================
ax3 = plt.subplot(2, 3, 3)
good_scores = [model.predict_score(h, 'Good')[0] for h in study_range]
bad_scores = [model.predict_score(h, 'Bad')[0] for h in study_range]

ax3.plot(study_range, good_scores, 'g-', linewidth=3, label='Good Mood')
ax3.plot(study_range, bad_scores, 'r-', linewidth=3, label='Bad Mood')
ax3.fill_between(study_range, good_scores, bad_scores, alpha=0.2, color='yellow')
ax3.set_xlabel('Study Hours', fontsize=11, fontweight='bold')
ax3.set_ylabel('Expected Exam Score', fontsize=11, fontweight='bold')
ax3.set_title('Expected Score vs Study Hours\n(Continuous given Continuous + Discrete)',
              fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 10)
ax3.set_ylim(40, 110)

# ============================================================
# 4. Scatter plot of samples colored by mood
# ============================================================
ax4 = plt.subplot(2, 3, 4)
samples = model.sample(n_samples=300)
good_samples = [s for s in samples if s['mood'] == 'Good']
bad_samples = [s for s in samples if s['mood'] == 'Bad']

ax4.scatter([s['study_hours'] for s in good_samples],
           [s['exam_score'] for s in good_samples],
           c='green', alpha=0.6, s=50, label='Good Mood', edgecolors='black', linewidth=0.5)
ax4.scatter([s['study_hours'] for s in bad_samples],
           [s['exam_score'] for s in bad_samples],
           c='red', alpha=0.6, s=50, label='Bad Mood', edgecolors='black', linewidth=0.5)

ax4.set_xlabel('Study Hours', fontsize=11, fontweight='bold')
ax4.set_ylabel('Exam Score', fontsize=11, fontweight='bold')
ax4.set_title('Sample Data (300 samples)\nColored by Mood',
              fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-0.5, 10.5)
ax4.set_ylim(40, 105)

# ============================================================
# 5. Distribution of Study Hours (continuous prior)
# ============================================================
ax5 = plt.subplot(2, 3, 5)
study_hours_samples = [s['study_hours'] for s in samples]
ax5.hist(study_hours_samples, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax5.axvline(x=np.mean(study_hours_samples), color='red', linestyle='--',
            linewidth=2, label=f'Mean = {np.mean(study_hours_samples):.2f}')
ax5.set_xlabel('Study Hours', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title('Distribution of Study Hours\n(Continuous Variable)',
              fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# ============================================================
# 6. Distribution of Exam Scores (continuous outcome)
# ============================================================
ax6 = plt.subplot(2, 3, 6)
exam_scores = [s['exam_score'] for s in samples]
ax6.hist(exam_scores, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
ax6.axvline(x=np.mean(exam_scores), color='blue', linestyle='--',
            linewidth=2, label=f'Mean = {np.mean(exam_scores):.2f}')
ax6.set_xlabel('Exam Score', fontsize=11, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax6.set_title('Distribution of Exam Scores\n(Continuous Variable)',
              fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# ============================================================
# Overall title and layout
# ============================================================
fig.suptitle('Heterogeneous Probabilistic Graphical Model Visualization\n' +
             'Mixing Discrete (Mood) and Continuous (Study Hours, Exam Score) Variables',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.985])

# Save the figure
output_file = '/home/user/personal-site/pgm_visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Visualization saved to: {output_file}")
print(f"📊 Generated {len(samples)} samples for visualization")

# Show statistics
print("\n📈 Statistics from samples:")
print(f"   Mean Study Hours: {np.mean(study_hours_samples):.2f} ± {np.std(study_hours_samples):.2f}")
print(f"   Mean Exam Score: {np.mean(exam_scores):.2f} ± {np.std(exam_scores):.2f}")
print(f"   % Good Mood: {len(good_samples)/len(samples)*100:.1f}%")
print(f"   Correlation (Hours vs Score): {np.corrcoef(study_hours_samples, exam_scores)[0,1]:.3f}")

plt.show()
