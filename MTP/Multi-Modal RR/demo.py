def objective(trial):
    global run_number  # Declare the global variable
    
    # Hyperparameter search space
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    
    # Update DataLoader with trial batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define model, loss, optimizer
    model = PyramidAttentionModel(input_channels=1, n_classes=6, num_splits=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"running epoch number:{epoch+1}")
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            gap_outputs, dct_outputs = model(inputs)
            loss_gap = criterion(gap_outputs, labels)
            loss_dct = criterion(dct_outputs, labels)
            loss = loss_gap + 2.5 * loss_dct
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    # Evaluation
    model.eval()

    correct_gap, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            gap_outputs, _ = model(inputs)
            _, predicted_gap = torch.max(gap_outputs, 1)
            total += labels.size(0)
            correct_gap += (predicted_gap == labels).sum().item()
    
    # Increment the run number for the next trial
    run_number += 1
    
    # Return accuracy for optimization
    accuracy = 100 * correct_gap / total
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1)
print("NAS completed")
# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)