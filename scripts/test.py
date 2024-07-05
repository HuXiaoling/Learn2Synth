import torch
from learn2synth.optim import SGD, Adam

matvec = lambda A, x: A.matmul(x.unsqueeze(-1)).squeeze(-1)
loss = lambda x, y: (x-y).square().mean()

torch.random.manual_seed(0)

U = torch.randn([4, 4], requires_grad=True)  # Segmentation net
R = torch.randn([4, 4], requires_grad=True)  # Residual synth net

optimU = Adam([U], lr=1e-3)
optimR = Adam([R], lr=1e-3)

synth, synth_truth = torch.randn([4]), torch.randn([4])
real, real_truth = torch.randn([4]), torch.randn([4])

# loop
for _ in range(3):

    optimU.zero_grad()
    optimR.zero_grad()

    # synth forward
    synth_better = synth + matvec(R, synth)
    synth_pred = matvec(U.clone(), synth_better)
    synth_loss = loss(synth_pred, synth_truth)
    synth_loss.backward(inputs=[U], create_graph=True)
    optimU.step()

    # real forward
    real_pred = matvec(U, real)
    real_loss = loss(real_pred, real_truth)
    real_loss.backward(inputs=[R])
    optimR.step()

foo = 0
