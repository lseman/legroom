"""Benchmark script to measure legroom compression performance."""
import time
import sys
sys.path.insert(0, '/home/seman/logician/legroom')

# Generate realistic test data
messages = []
messages.append({"role": "system", "content": "You are a helpful assistant."})
messages.append({"role": "user", "content": "Analyze these search results for 'machine learning optimization techniques' and summarize the key findings."})

for i in range(50):
    parts = []
    parts.append("Search result " + str(i) + ":")
    parts.append("Title: " + ("Optimization Technique " + str(i) + " ") * 5)
    parts.append("Abstract: This paper explores advanced machine learning optimization techniques including gradient descent variants, learning rate scheduling, and adaptive moment estimation. We demonstrate that combining AdamW with cyclical learning rates and warmup schedules yields 15-30% improvement in convergence speed on transformer models.")
    for j in range(10):
        parts.append("Key findings: Method A achieved " + str(i * 0.1 * j) + "% accuracy.")
    parts.append("Method B showed " + str(i * 0.05) + "% improvement on validation set.")
    parts.append("References: [1] Smith et al., 2024. [2] Johnson & Lee, 2023. [3] Wang et al., 2024.")
    messages.append({"role": "user", "content": "\n".join(parts)})

code_snippet = """def optimize(model, data, lr=0.001, epochs=100):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for batch in data:
            optimizer.zero_grad()
            loss = model(batch).loss()
            loss.backward()
            optimizer.step()
            scheduler.step()
        if loss < best_loss:
            best_loss = loss
    return best_loss"""

for i in range(30):
    parts = ["Turn " + str(i) + ": Based on search results " + str(i*2) + "-" + str(i*2+2) + "."]
    parts.append("The key optimization technique that stands out is **adaptive learning rate scheduling**.")
    parts.append("```python")
    parts.append(code_snippet)
    parts.append("```")
    parts.append("This implementation uses AdamW with cosine annealing warm restarts.")
    messages.append({"role": "assistant", "content": "\n".join(parts)})

for i in range(20):
    parts = ["Turn " + str(i) + ": Here's my analysis of the optimization techniques."]
    parts.append("The evidence strongly suggests that **adaptive learning rate methods** combined with **gradient clipping** provide the most consistent improvements across different model architectures.")
    parts.append("Additionally, the combination of weight decay with adaptive methods (as in AdamW) appears to be critical for generalization.")
    messages.append({"role": "assistant", "content": "\n".join(parts)})

from legroom import compress
from legroom.compressors.balanced_end import _HAS_NUMBA

# Warm up
compress(messages[:5], model="gpt-4o")

# Benchmark (5 runs)
times = []
for run in range(5):
    t0 = time.perf_counter()
    result = compress(messages, model="gpt-4o")
    t1 = time.perf_counter()
    times.append(t1 - t0)

avg_time = sum(times) / len(times)

print("=" * 60)
print("LEGROOM COMPRESSION BENCHMARK")
print("=" * 60)
print("Messages: {}".format(len(messages)))
print("Tokens before: {:,}".format(result.tokens_before))
print("Tokens after: {:,}".format(result.tokens_after))
print("Compression ratio: {:.2%}".format(result.tokens_after / result.tokens_before))
print("Tokens saved: {:,} ({:.1f}%)".format(
    result.tokens_saved,
    (1 - result.tokens_after / result.tokens_before) * 100
))
print("Transforms applied:", result.transforms_applied)
print("Warnings:", result.warnings)
print("=" * 60)
print("Performance:")
print("  Time (5 runs):")
for t in times:
    print("    {:.3f}s".format(t))
print("  Average: {:.3f}s".format(avg_time))
print("  Tokens/sec: {:,}".format(int(result.tokens_before / avg_time)))
print("  Numba available: {}".format(_HAS_NUMBA))
print("=" * 60)
