import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# 1. The Tickets
tickets = [
    "I forgot my password, how to reset it?",
    "I can't log in, as password is incorrect.",
    "How to see leave balance ?"
]

print("Loading AI Model... (this might take a moment)")
# 2. Load Pre-trained Embedding Model
# 'all-MiniLM-L6-v2' is a small, fast model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Convert Tickets to Vectors (Embeddings)
embeddings = model.encode(tickets)

print(f"Encoded {len(tickets)} tickets into vectors of size {embeddings.shape[1]}.\n")

# 4. Group the Tickets (Clustering)
# We want 2 groups: Access Issues vs. HR Issues
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
kmeans.fit(embeddings)

# 5. Display the Results
cluster_labels = kmeans.labels_

# Organize by cluster
grouped_tickets = {}
for i, label in enumerate(cluster_labels):
    if label not in grouped_tickets:
        grouped_tickets[label] = []
    grouped_tickets[label].append(tickets[i])

print("--- AI Groping Results ---")
for cluster_id, ticket_list in grouped_tickets.items():
    print(f"\nGroup {cluster_id + 1}:")
    for ticket in ticket_list:
        print(f" - {ticket}")

print("\n--------------------------")
print("Done! The AI has successfully grouped the tickets.")
