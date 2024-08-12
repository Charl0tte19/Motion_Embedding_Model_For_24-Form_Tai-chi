import numpy as np
import matplotlib.pyplot as plt

duration = [11, 24, 7, 26, 6, 29, 24, 26, 12, 21, 8, 7, 10, 7, 10, 14, 14, 16, 7, 6, 14, 8, 9, 9]
dataset_embs = {form_id: {clip_id: None for clip_id in range(duration[form_id])} for form_id in range(24)}
teacher_embs = {form_id: {clip_id: None for clip_id in range(duration[form_id])} for form_id in range(24)}

similarities = []
form_clips = []

for form_id in dataset_embs.keys():
    for clip_id in dataset_embs[int(form_id)].keys():
        dataset_embs[int(form_id)][clip_id] = np.load(f'./Taichi_Clip/forms_keypoints/dataset_embs/{form_id}_{clip_id}.npy')
        teacher_embs[int(form_id)][clip_id] = np.load(f'./Taichi_Clip/teacher_keypoints/teacher_embs/{form_id}_{clip_id}.npy')

        dot_product = np.dot(dataset_embs[int(form_id)][clip_id], teacher_embs[int(form_id)][clip_id])

        norm_a = np.linalg.norm(dataset_embs[int(form_id)][clip_id])
        norm_b = np.linalg.norm(teacher_embs[int(form_id)][clip_id])

        cosine_similarity = dot_product / (norm_a * norm_b)

        similarity = (cosine_similarity + 1) / 2

        similarities.append(similarity)
        form_clips.append(f"{form_id}_{clip_id}")

fig, ax = plt.subplots(figsize=(15, 6))

bars = ax.bar(form_clips, similarities, color='#DFC57B')

min_value = min(similarities)
min_index = np.argmin(similarities)

ax.axhline(y=min_value, color='black', linestyle='--')
ax.text(len(form_clips)+1, min_value + 0.02, f'{min_value:.2f}', ha='left', va='center', color='black')

ax.set_xticks([])

plt.xlabel('Form Clip')
plt.ylabel('Similarity')
plt.title('Similarity of Form Clips')
plt.show()