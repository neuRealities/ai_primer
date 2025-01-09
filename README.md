<a name="readme-top" id="readme-top"></a>
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
[![Logo][logo]][logo-url]
<br />
<div align="center">
<h1 align="center">AI Primer</h1>

  <p align="center">
    A quick start installation of AI tools to get you started with your first AI projects.
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This primer project offers a streamlined path for AI tool installation, to quick start Data Scientist tools like Jupyter notebooks.

You can optimize performance with GPU-support, but that's beyond the scope of this project. For Docker/CUDA support, you can search online, or follow IQXR's Accelerated Engineering `prototype-base` [Azure repository](https://dev.azure.com/iqxr/PlatformCapabilities/_git/prototype-base).


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Docker][Docker]][Docker-url]
* [![Jupyter][Jupyter]][Jupyter-url]
* [![Python][Python]][Python-url]
* [![NumPy][NumPy]][NumPy-url]
* [![PyTorch][PyTorch]][PyTorch-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

Getting a data scientist setup to work requires coordination between platform and app resources. The following elements are needed:

- **WSL**: Windows Subsystem for Linux - Allows Linux on Windows without dualboot.
- **Docker**: Platform designed to help developers build, share, and run container applications and their setup. 
- **Jupyter**: An open-source web application to create and share computational documents that combine code and rich text elements.

<!-- INSTALLATION -->
## Installation

### WSL

Let's install virtualization on your Windows machine. Open the command tool with admin privileges. Below is the shortcut way:

- Press `Windows` + `X`,
- Then `A` For Terminal **A**dmin,
- Allow changes if notification pops up.

In the console, type

```bash
wsl --install
```

### Docker 

Install Docker Desktop on Windows

1. Following the [setup link](https://docs.docker.com/desktop/setup/install/windows-install/).
2. Select WSL 2 if asked.
3. Restart machine at the end.
4. Enable Docker on WSL 2 by following [these instructions](https://docs.docker.com/desktop/features/wsl/)

### Git

Lets copy this repository to your WS Ubuntu setup, so that we can run Jupyter in your recently installed Docker.

1. Start Docker Desktop
2. Start Ubuntu in your Windows machine (There should be a new app icon installed)
3. Clone this repo by typing `git clone https://github.com/IQXR/ai_primer` 
4. `cd ai_primer`
5. `docker compose up`
6. Wait for docker to finish initial setup
7. `Ctrl` + `click` on the displayed webserver address. It will likely be [http://127.0.0.1:8888/lab/]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

1. On the left side navigation, click on `notebooks`
2. Open the Jupyter notebook namesd `00-Intro`
3. You're done installing!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- Get new badges at [https://github.com/Ileriayo/markdown-badges] -->
[logo]: media/logo/Logo_GRADIENT_H_KO.png
[logo-url]: https://www.iqxr.com/

[contributors-shield]: https://img.shields.io/badge/Contributors-Accelerated_Engineering-red?style=for-the-badge
[contributors-url]: https://iqxr.atlassian.net/wiki/spaces/AE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/company/iqxr/

[Blender]: https://img.shields.io/badge/blender-%23F5792A.svg?style=for-the-badge&logo=blender&logoColor=white
[Blender-url]: https://www.blender.org/
[Docker]: https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white
[Docker-url]: https://www.docker.com/
[Git]: https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white
[Git-url]: https://git-scm.com/ 
[GitHub]: https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white
[GitHub-url]: https://github.com/
[Jupyter]: https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white
[Jupyter-url]: https://jupyter.org/
[Matplotlib]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Matplotlib-url]: https://matplotlib.org/
[NumPy]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[nVIDIA]: https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white
[nVIDIA-url]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html
[OpenCV]: https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white
[OpenCV-url]: https://opencv.org/
[Pandas]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/
[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[PyTorch]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[SciPy]: https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white
[SciPy-url]: https://scipy.org/
[TensorFlow]: https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white
[TensorFlow-url]: https://www.tensorflow.org/
