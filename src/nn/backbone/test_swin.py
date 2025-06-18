if __name__ == "__main__":
    # Register the Swin model
    from Swin import register
    register(Swin)

    # Create example YAML files
    os.makedirs('test_configs', exist_ok=True)
    with open('test_configs/swin_config.yml', 'w') as f:
        f.write("""
type: Swin
variant: b
return_idx: [1, 2, 3, 4]
freeze_at: -1
freeze_norm: False
pretrained: True
""")

    # Load and merge configurations
    swin_config = load_config('test_configs/swin_config.yml')

    # Create an instance of Swin
    swin_instance = create('Swin', **swin_config)
    print(f"Created Swin instance with configuration: {swin_config}")

    # Test the forward pass (example input)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    input_tensor = torch.randn(1, 3, 224, 224).to(device)  # Example input
    output = swin_instance(input_tensor)
    print(f"Output from Swin: {output}")