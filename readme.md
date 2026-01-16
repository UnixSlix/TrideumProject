# Running the application

1. Navigate to the **TrideumProject/app** folder.
2. User docker compose to start the build process.
    - **To run without using GPU**  
    ```docker compose up --build``` **MUCH SLOWER LLM RESPONSES, HIGHER DRAM USAGE**
    - **To use GPU (Windows only with WSL2 configured with Nvidia Container Toolkit installed)**  
        ```docker compose -f compose.yaml -f compose.gpu.yaml up --build```
    - **Optional compose flag to enable Watch (live rebuilds on source changes for rag_app container)**  
    ```--watch```

# Ollama Container GPU Support on Windows  

To take advantage of an Nvidia GPU for the Ollama container on a Windows machine, the host must have the Nvidia Container Toolkit installed.  

## Prerequisites  

1. **Windows Subsystem for Linux 2 (WSL2):**  
    - Ensure WSL2 is installed and running on your Windows machine.  
    - By default, the minimum WSL2 distro will be running, but you need to install a version of Ubuntu (22.04 is the recommended version at the time of writing).  

2. **Nvidia Container Toolkit:**  
    - Install the Nvidia Container Toolkit in the Linux distribution running in WSL2.  
    - This is required for Docker Desktop on Windows to access the GPU.  

## Important Notes  

- The Nvidia Container Toolkit is a dependency for GPU access to containers for the Docker engine.  
- It **cannot** be installed in the container via the build process.  
- The toolkit must be preinstalled on the host machine.  

For more information, refer to the official Nvidia Container Toolkit documentation.  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
Enabling Docker Compose GPU access https://docs.docker.com/compose/how-tos/gpu-support/

# Text file sample data sets for testing

A Natural Language Processing focused repository of free/public domain datasets all in .txt. https://github.com/niderhoff/nlp-datasets?utm_source=chatgpt.com