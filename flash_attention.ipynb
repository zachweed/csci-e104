{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# Zachary Weed\n"
      ],
      "metadata": {
        "id": "KtMIPg_sFv63"
      },
      "id": "KtMIPg_sFv63"
    },
    {
      "cell_type": "code",
      "source": [
        "pip install flash-attn --no-build-isolation -v"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BFLdsZ0qy0Qo",
        "outputId": "ffa8d000-f6ca-45f2-f381-19fdda57680c"
      },
      "id": "BFLdsZ0qy0Qo",
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Using pip 24.1.2 from /usr/local/lib/python3.12/dist-packages/pip (python 3.12)\n",
            "Requirement already satisfied: flash-attn in /usr/local/lib/python3.12/dist-packages (2.8.3)\n",
            "Requirement already satisfied: torch in /usr/local/lib/python3.12/dist-packages (from flash-attn) (2.10.0+cu128)\n",
            "Requirement already satisfied: einops in /usr/local/lib/python3.12/dist-packages (from flash-attn) (0.8.2)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (3.29.0)\n",
            "Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (4.15.0)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (75.2.0)\n",
            "Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (1.14.0)\n",
            "Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (3.6.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (3.1.6)\n",
            "Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (2025.3.0)\n",
            "Requirement already satisfied: cuda-bindings==12.9.4 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (12.9.4)\n",
            "Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.8.93 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (12.8.93)\n",
            "Requirement already satisfied: nvidia-cuda-runtime-cu12==12.8.90 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (12.8.90)\n",
            "Requirement already satisfied: nvidia-cuda-cupti-cu12==12.8.90 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (12.8.90)\n",
            "Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (9.10.2.21)\n",
            "Requirement already satisfied: nvidia-cublas-cu12==12.8.4.1 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (12.8.4.1)\n",
            "Requirement already satisfied: nvidia-cufft-cu12==11.3.3.83 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (11.3.3.83)\n",
            "Requirement already satisfied: nvidia-curand-cu12==10.3.9.90 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (10.3.9.90)\n",
            "Requirement already satisfied: nvidia-cusolver-cu12==11.7.3.90 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (11.7.3.90)\n",
            "Requirement already satisfied: nvidia-cusparse-cu12==12.5.8.93 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (12.5.8.93)\n",
            "Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (0.7.1)\n",
            "Requirement already satisfied: nvidia-nccl-cu12==2.27.5 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (2.27.5)\n",
            "Requirement already satisfied: nvidia-nvshmem-cu12==3.4.5 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (3.4.5)\n",
            "Requirement already satisfied: nvidia-nvtx-cu12==12.8.90 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (12.8.90)\n",
            "Requirement already satisfied: nvidia-nvjitlink-cu12==12.8.93 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (12.8.93)\n",
            "Requirement already satisfied: nvidia-cufile-cu12==1.13.1.3 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (1.13.1.3)\n",
            "Requirement already satisfied: triton==3.6.0 in /usr/local/lib/python3.12/dist-packages (from torch->flash-attn) (3.6.0)\n",
            "Requirement already satisfied: cuda-pathfinder~=1.1 in /usr/local/lib/python3.12/dist-packages (from cuda-bindings==12.9.4->torch->flash-attn) (1.5.3)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch->flash-attn) (1.3.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch->flash-attn) (3.0.3)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "id": "848196d2-0628-418e-a562-4836c1f26367",
      "metadata": {
        "id": "848196d2-0628-418e-a562-4836c1f26367"
      },
      "source": [
        "# Head Class"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "id": "5087f6f4-6cf9-4ea7-b84f-253cb8eb9df9",
      "metadata": {
        "id": "5087f6f4-6cf9-4ea7-b84f-253cb8eb9df9"
      },
      "outputs": [],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "\n",
        "from torch.nn import functional as F\n",
        "from flash_attn import flash_attn_func\n",
        "\n",
        "class Head(nn.Module):\n",
        "  def __init__(self, head_size, attention_type, number_of_embeddings = 32, block_size = 8):\n",
        "    super().__init__()\n",
        "\n",
        "    # compute projections from embeddings\n",
        "    # into a different space, as:\n",
        "    # Query, Key, and Value\n",
        "    self.query = nn.Linear(number_of_embeddings, head_size, bias=False)\n",
        "    self.key = nn.Linear(number_of_embeddings, head_size, bias = False)\n",
        "    self.value = nn.Linear(number_of_embeddings, head_size, bias = False)\n",
        "    self.attention_type = attention_type\n",
        "\n",
        "    # registers causal mask on  model\n",
        "    self.register_buffer('tril', self.build_causal_mask(block_size))\n",
        "\n",
        "  # Ensures attention is\n",
        "  # only maintained for tokens\n",
        "  # that have already been processed.\n",
        "  # The tensor is essentially like this:\n",
        "  #   a | b | c\n",
        "  # a 1 | 0 | 0\n",
        "  # b 1 | 1 | 0\n",
        "  # c 1 | 1 | 1\n",
        "  # Each row represents what\n",
        "  # that token can attend with, so:\n",
        "  #   1) 'a' can attend with a, but not b & c,\n",
        "  #   2) 'b' can attend with a, and b, but not c, &\n",
        "  #   3) 'c' can attend with all\n",
        "  #\n",
        "  # Step-by-step\n",
        "  #\n",
        "  # With respect to time, if at step one, we are processing\n",
        "  # row number 1, and thus token 'a', then:\n",
        "  #   a | b | c\n",
        "  # a 1 | 0 | 0 present\n",
        "  # b 1 | 1 | 0 future\n",
        "  # c 1 | 1 | 1 future\n",
        "  #\n",
        "  # Then as it shifts to token 'b', then:\n",
        "  #   a | b | c\n",
        "  # a 1 | 0 | 0 past\n",
        "  # b 1 | 1 | 0 present\n",
        "  # c 1 | 1 | 1 future\n",
        "  #\n",
        "  # Then as it shifts to token 'c':\n",
        "  #   a | b | c\n",
        "  # a 1 | 0 | 0 past\n",
        "  # b 1 | 1 | 0 past\n",
        "  # c 1 | 1 | 1 present\n",
        "  #\n",
        "  # Thus, if we are at token 'a' in step one,\n",
        "  # then 'a' can map with its self\n",
        "  # and no other tokens. And if we are at\n",
        "  # token 'b' in step two, then it can map with\n",
        "  # a 'past token' ('a'), and its self\n",
        "  #\n",
        "  def build_causal_mask(self, block_size):\n",
        "    square_matrix = torch.ones(block_size, block_size)\n",
        "    triangular_matrix = torch.tril(square_matrix)\n",
        "    return triangular_matrix\n",
        "\n",
        "\n",
        "  def forward(self, x):\n",
        "    if self.attention_type == \"traditional_attention_forward\":\n",
        "      return self.traditional_attention_forward(x)\n",
        "    else:\n",
        "      return self.flash_attention_forward(x)\n",
        "\n",
        "  def traditional_attention_forward(self, x):\n",
        "    batch, time, channels = x.shape\n",
        "\n",
        "    # matrix multiplication of keys and queries\n",
        "    # representing what is contained, and what\n",
        "    # is being looked for, respectively.\n",
        "    q = self.query(x)\n",
        "    k = self.key(x)\n",
        "\n",
        "    attention_score_matrix = q @ k.transpose(-2, -1) * channels ** -0.5\n",
        "\n",
        "    # since we have the triangular matrix, as a causal mask,\n",
        "    # here we map the mask so that anything with value of 0\n",
        "    # (not allowed) becomes mapped as True, and anything\n",
        "    # with a value of 1 becomes mapped as False.\n",
        "    # The -inf is used as a means of swapping\n",
        "    # True with -inf so the softmax() function provides 0\n",
        "    # probability when applying the causal mask\n",
        "    #\n",
        "    # Thus, if we have\n",
        "    #   a | b | c |\n",
        "    # a 1 | 0 | 0 |\n",
        "    # b 1 | 1 | 0 |\n",
        "    # c 1 | 1 | 1 |\n",
        "    #\n",
        "    # Then it becomes the vector below,\n",
        "    # before transforming with -inf\n",
        "    # (with F means allowed and T means disallowed)\n",
        "    #\n",
        "    #   a | b | c |\n",
        "    # a F | T | T |\n",
        "    # b F | F | T |\n",
        "    # c F | F | F |\n",
        "    #\n",
        "    # Transforming with -inf produces\n",
        "    #   a | b    | c    |\n",
        "    # a F | -inf | -inf |\n",
        "    # b F | F    | -inf |\n",
        "    # c F | F    | F    |\n",
        "    #\n",
        "    attention_score_matrix = attention_score_matrix.masked_fill(\n",
        "      self.tril[:time, :time] == 0, float('-inf')\n",
        "    )\n",
        "\n",
        "    # computes aforementioned softmax, now that\n",
        "    # values are properly standardized\n",
        "    # The softmax makes sure that anything\n",
        "    # with -inf has a 0 probability\n",
        "    #   a | b | c |\n",
        "    # a F | 0 | 0 |\n",
        "    # b F | F | 0 |\n",
        "    # c F | F | F |\n",
        "    attention_score_matrix = F.softmax(attention_score_matrix, dim = -1)\n",
        "    return attention_score_matrix @ self.value(x)\n",
        "\n",
        "  # Essentially identical in implementation with the original Flashattention function\n",
        "  def flash_attention_forward(self, x):\n",
        "    q = self.query(x)\n",
        "    k = self.key(x)\n",
        "    v = self.value(x)\n",
        "\n",
        "    # so we can provide another dimension\n",
        "    q = q.unsqueeze(2)\n",
        "    k = k.unsqueeze(2)\n",
        "    v = v.unsqueeze(2)\n",
        "\n",
        "    # removes the additional dimension on output so we can verify similarity\n",
        "    return flash_attn_func(q, k, v, causal = True).squeeze(2)"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "ee167f4c-2f8f-4038-ab76-935acdec5fad",
      "metadata": {
        "id": "ee167f4c-2f8f-4038-ab76-935acdec5fad"
      },
      "source": [
        "# Multihead Class"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "id": "845c3c31-3f1e-4713-b4eb-32a26130f014",
      "metadata": {
        "id": "845c3c31-3f1e-4713-b4eb-32a26130f014"
      },
      "outputs": [],
      "source": [
        "class MultiHeadAttention(nn.Module):\n",
        "  def __init__(self, number_of_heads, head_size):\n",
        "    super().__init__()\n",
        "    self.heads = nn.ModuleList(\n",
        "      [\n",
        "        Head(head_size) for _ in range(number_of_heads)\n",
        "      ]\n",
        "    )\n",
        "\n",
        "  def forward(self, x):\n",
        "    return torch.cat([head(x) for head in self.heads], dim = -1)"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# CUDA Benchmarking Cell"
      ],
      "metadata": {
        "id": "3ACGExZFS8lQ"
      },
      "id": "3ACGExZFS8lQ"
    },
    {
      "cell_type": "code",
      "source": [
        "def benchmark(fn, x, warmup = 20, iterations = 100):\n",
        "  for _ in range(warmup):\n",
        "    _ = fn(x)\n",
        "  torch.cuda.synchronize()\n",
        "\n",
        "  start = torch.cuda.Event(enable_timing = True)\n",
        "  end = torch.cuda.Event(enable_timing = True)\n",
        "\n",
        "  start.record()\n",
        "\n",
        "  for _ in range(iterations):\n",
        "    _ = fn(x)\n",
        "\n",
        "  end.record()\n",
        "  torch.cuda.synchronize()\n",
        "\n",
        "  total_milliseconds = start.elapsed_time(end)\n",
        "\n",
        "  return total_milliseconds / iterations\n",
        "\n"
      ],
      "metadata": {
        "id": "_gr8fIBwS_rG"
      },
      "id": "_gr8fIBwS_rG",
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Calling Traditional and Flashattention and Verifying Similarity"
      ],
      "metadata": {
        "id": "hEQ6h87mPO_A"
      },
      "id": "hEQ6h87mPO_A"
    },
    {
      "cell_type": "code",
      "source": [
        "device = \"cuda\"\n",
        "data_type = torch.float16\n",
        "\n",
        "batch_size = 2\n",
        "time = 8\n",
        "channels = 32\n",
        "head_size = 32\n",
        "\n",
        "# Builds head\n",
        "head = Head(\n",
        "  head_size = head_size,\n",
        "  attention_type = \"traditional_attention_forward\",\n",
        "  number_of_embeddings = channels,\n",
        "  block_size = time\n",
        ").to(device).to(data_type)\n",
        "\n",
        "# Builds X for testing\n",
        "x = torch.randn(batch_size, time, channels, device = device, dtype = data_type)\n",
        "\n",
        "head.eval()\n",
        "\n",
        "with torch.no_grad():\n",
        "  # Computes with traditional architecture\n",
        "  traditional = benchmark(head.traditional_attention_forward, x)\n",
        "  # Computes with flash architecture\n",
        "  flash = benchmark(head.flash_attention_forward, x)\n",
        "\n",
        "\n",
        "# Identifies that the Results are essentially equivalent\n",
        "print(f\"Traditional: {traditional}\")\n",
        "print(f\"Flash: {flash}\")\n",
        "print(f\"How Much Faster: {traditional/flash:.2f}x\")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XJhuUhZUPTEL",
        "outputId": "af2037fc-8f7d-45f3-fa7b-557c3de90f63"
      },
      "id": "XJhuUhZUPTEL",
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Traditional: 0.29715456008911134\n",
            "Flash: 0.26838016510009766\n",
            "How Much Faster: 1.11x\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "deepseek",
      "language": "python",
      "name": "deepseek"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.11.15"
    },
    "colab": {
      "provenance": [],
      "gpuType": "A100"
    },
    "accelerator": "GPU"
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
