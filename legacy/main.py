def main():
    print("Hello from video-blade!")
    import torch
    x = torch.rand(5, 3).to("cuda")
    print(x)
    print("Done!")


if __name__ == "__main__":
    main()
