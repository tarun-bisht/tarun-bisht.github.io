---
layout: writing
title:  "how i set up my arch linux system for development and machine learning"
date:   2021-08-13 13:42:00 +0530
category: Linux
tags: linux arch-linux manjaro arch-setup ml-setup
description: In this post, I am sharing how I set up my Arch Linux system for machine learning and development. If I boot up a fresh arch install I do these steps and set up the machine for my work and productivity. There are very few guides to set up a development and machine learning arch Linux based systems and I hope this post will be helpful to someone setting up an arch Linux working environment.
comment: true
math: false
---

In this post, I am sharing how I set up my Arch Linux system for machine learning and development. If I boot up a fresh arch install I do these steps and set up the machine for my work and productivity. There are very few guides to set up a development and machine learning arch Linux based environment and I hope this post will guide you to set up your Arch Linux working environment. Softwares and extensions which I installed in this post can be tweaked based on your needs and preferences. All the below steps works well for me. If you are starting your Arch journey then you can start with Manjaro which is an arch based distro which I am personally using as my daily device.


## Why Arch Linux
- Most technologies somehow utilize linux in their development cycle example setting up a cloud server, web server etc. so learning linux will give you an upper hand. Arch Linux lets you to better understand how Linux works. 
- Arch Linux is a rolling release distribution that means the new kernel version or application version rolls out as soon as they are released.
- The most convincing reason is AUR (arch user repository) which mostly have any application that one might think of and easy to install. This feature distinguishes arch Linux from other Linux distros.
- It is way more performant than windows and ubuntu, if the lightweight desktop environment is used like Xfce then it also consume less RAM.


## Update Mirrors
{% highlight bash linenos %}
sudo pacman-mirrors --fasttrack
{% endhighlight %}

## Update System
{% highlight bash linenos %}
sudo pacman -Syyu
{% endhighlight %}

## Install yay
yay(yet another yogurt) is an AUR helper that lets to install a new package from AUR, update and remove installed AUR packages.
{% highlight bash linenos %}
pacman -S --needed base-devel
git clone https://aur.archlinux.org/yay.git
cd yay
makepkg -si
cd ..
rm -rf yay
{% endhighlight %}

## Install auto-cpufreq
If you are in power restricted devices like laptops, Linux might not give the battery backup that you expected. Most CPUs are not power-optimized for Linux, installing this package will automatically adjust CPU speed & power based on usage. It allows us to improve battery life without making any compromises. We do not have to configure anything just install it and see the difference. It's a must install tool on my laptop. 

{% highlight bash linenos %}
sudo pacman -S auto-cpufreq
systemctl enable auto-cpufreq
systemctl start auto-cpufreq
{% endhighlight %}

## Install necessary software
Now install all your productivity applications, favourite applications, IDE's, code editors, Spotify etc. With the power of AUR, all these things are in your reach, no need to hop around just search AUR and find your application.

In fresh installation, I install these applications:
- chrome
- brave browser
- slack
- discord
- kde-connect
- vscode
- vim
- miniconda

{% highlight bash linenos %}
yay -S chrome brave-bin slack-desktop discord spotify kdeconnect vim
{% endhighlight %}

I install miniconda from its website instead of AUR as it already set up all things automatically also miniconda package I once installed from AUR was conflicting with python installed.

{% highlight bash linenos %}
cd Downloads
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sudo chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
{% endhighlight %}

now proceed to the installation process and now you have conda setup just restart the terminal and check conda version if its working or not. By default installing this way will add the path to the `.bashrc` file.

{% highlight bash linenos %}
conda --version
{% endhighlight %}

## Setting up VSCode
Installing vscode

{% highlight bash linenos %}
sudo pacman -S vscode
{% endhighlight %}

By default I could not access vscode marketplace for extensions it always says *we cannot connect to extension marketplace*. I installed code-marketplace package from AUR and it fixed the issue.

{% highlight bash linenos %}
yay -S code-marketplace
{% endhighlight %}

Next, I install language extensions for C, C++, Python and Rust other language extensions I add based on the use case.

Next, I install my favourite vscode extensions for productivity, these extensions I add anytime when setting up freshly installed vscode. 
- error lens
- auto rename tag
- git lens
- live sass compiler
- live share
- prettier
- css-auto-prefix
- live server
- bracket pair colorizer
- arepl
- better comments
- dash
- python docstring generator

## Setting up conda
With the installation of miniconda, when we open a new instance of the terminal base environment is activated by default which I do not like. I want to activate conda environment when I needed them. We can disable this default behaviour with the following command.

{% highlight bash linenos %}
disable default activation of base environment: conda run conda config --set auto_activate_base false
{% endhighlight %}

## Machine Learning setup
### Installing nvidia drivers
- First install nvidia latest drivers using
{% highlight bash linenos %}
sudo pacman -S nvidia nvidia-utils
{% endhighlight %}
If you are using manjaro with proprietory drivers then no need to do the above step as Nvidia drivers are already installed.

- Check nvidia drivers are installed correctly and version 
{% highlight bash linenos %}
nvidia-smi
{% endhighlight %}
### Instaling PyTorch and TensorFlow
We can install packages using pip or conda based on preference. PyTorch has very good installation instruction and is easy to setup just use latest installation instruction from its [website](https://pytorch.org/get-started/locally/) for conda and pip. For TensorFlow we can use conda which will install all dependencies along with cuda and cudnn but we might not get latest version of library. I prefer installing TensorFlow using pip because conda TensorFlow packare is outdated and I was not able to create tensors in GPU while creating tensors the process gets stuck without any errors. Installing TensorFlow via pip require some additional steps.

- Create an environment using conda:

{% highlight bash linenos %}
conda create --name envname
{% endhighlight %}

- Activate created environment

{% highlight bash linenos %}
conda activate envname
{% endhighlight %}

#### PyTorch CUDA setup

{% highlight bash linenos %}
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
{% endhighlight %}
for latest instructions visit [pytorch](https://pytorch.org/get-started/locally/) official site.

#### TensorFlow CUDA setup
**Using conda**

{% highlight bash linenos %}
conda install tensorflow-gpu 
{% endhighlight %}

**Using pip**

With this method we have to also install `cuda-toolkit` and `cudnn` these packages can be found in pacman.

{% highlight bash linenos %}
sudo pacman -S cuda-toolkit cudnn
{% endhighlight %}

After installing restart the system and then verify cuda install

{% highlight bash linenos %}
nvcc -V
{% endhighlight %}

Now activate virtual conda environment and install Tensorflow using pip

{% highlight bash linenos %}
conda install pip
pip install tensorflow-gpu
{% endhighlight %}

#### Verify cuda detection by PyTorch and TensorFlow

We can verify detection of cuda by TensorFlow and PyTorch using a small python code snippet and check its output

**PyTorch**

{% highlight python linenos %}
import torch
print(torch.cuda.is_available()) # should be True
{% endhighlight %}

**TensorFlow**

{% highlight bash linenos %}
import tensorflow as tf
print(tf.config.list_physical_devices("GPU")) # should be list of GPUS
{% endhighlight %}

## Jekyll setup
- install ruby
{% highlight bash linenos %}
sudo pacman -S ruby
{% endhighlight %}
- install jekyll
{% highlight bash linenos %}
gem install jekyll
{% endhighlight %}
- export gems path to bashrc
open .bashrc file and add these lines at last
{% highlight bash linenos %}
export GEM_HOME="/home/username/.local/share/gem/ruby/3.0.0/gems/"
export PATH="/home/username/.local/share/gem/ruby/3.0.0/bin/:$PATH"
{% endhighlight %}

## Installing Rust
Rust is installed and managed by the rustup tool it makes the process easier to install and manage rust versions.
{% highlight bash linenos %}
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
{% endhighlight %}

## Setting up GitHub ssh
Github has now removed support for password-based authentication for git operations, so accessing git through the command line will not be possible after August 2021. Github will no longer accept account passwords when authenticating Git operations on GitHub.com. We have to use a token-based authentication or access via ssh. I use ssh based authentication for git operations in Github.com.

- generating public private key pair
{% highlight bash linenos %}
- ssh-keygen -t ed25519 -C "your email here"
{% endhighlight %}
- upload public key to github > settings > ssh and gpg keys. Public key will have pub extension just copy the contents of that file and paste in the textbox provided.

- adding ssh key to ssh-agent
{% highlight bash linenos %}
ssh-add .ssh/generated_keyname
{% endhighlight %}
- test connection is working fine or not
{% highlight bash linenos %}
ssh -T git@github.com
{% endhighlight %}
it will print out your username in console.

## Change time format to RTC (for dual boot users with windows)
If you are using arch desktop in dual boot settings with windows then time can change if you hop between different boots. The below command will enable RTC convention instead of UTC for time and date in linux which windows uses by default. 
{% highlight bash linenos %}
timedatectl set-local-rtc 1 --adjust-system-clock
{% endhighlight %}

## Auto Mount Hard drives or partitions
If you have system with more than one hard drives, like in my case I have two hard drives a M.2 SSD in which linux is installed and a mechanical hard drive for backup and media contents. By default linux will not mount hard drive or partitions automatically other than swap, root and home, so some applications that rely on these drives for saving and loading data will fail on startup as they will not able to access that location. I usually auto mount my backup and media drives to load my music automatically configure backup at startup.

- Create a folder where we want to mount the drive. In my case I want to mount my drives in folder named *media*
{% highlight bash linenos %}
sudo mkdir /media
{% endhighlight %}
- Create subfolders inside that folder to mount multiple partitions.
{% highlight bash linenos %}
sudo mkdir /media/multimedia
sudo mkdir /media/backup
{% endhighlight %}
I want to mount my backup drive in /media/backup and multimedia drive in /media/multimedia
- Check disk and partitions in system
{% highlight bash linenos %}
sudo fdisk -l
{% endhighlight %}
- Find UUID of drive which we want to auto mount
{% highlight bash linenos %}
sudo blkid
{% endhighlight %}
- Copy UUID's of required drives for auto mounting
- Open fstab file and add entry for drives
{% highlight bash linenos %}
sudo nano /etc/fstab
{% endhighlight %}
The entry we have to add in the file should be of form 
```bash
UUID="PASTE HERE"<tab>/media/folder_name<tab>filesystem<tab>defaults<tab>0<tab>0
```
here is an example
{% highlight bash linenos %}
UUID=01D4653A0E0898E0	/media/multimedia	ext4	defaults	0	0
{% endhighlight %}
For more info about editing fstab file and details of different parameters visit [this link](https://www.howtogeek.com/444814/how-to-write-an-fstab-file-on-linux/)

If you are mounting windows ntfs drive, it might not give write permissions to that drive, to solve this issue we have to follow some additional steps.

- Disable fast startup in windows
The option can be found inside `Control Panel > Power Options > Choose what the power button do > Turn off fast startup`

- Install ntfs-3g
{% highlight bash linenos %}
sudo pacman -S ntfs-3g
{% endhighlight %}

- Find your ntfs drive to enable write access
{% highlight bash linenos %}
sudo blkid
{% endhighlight %}
this command lists all drive partitions in the device, from this select the ntfs drive to enable write access.

- Enable write access
{% highlight bash linenos %}
sudo ntfsfix /dev/sda3
{% endhighlight %}
replace `/dev/sda3` with your partition.

We can also automount windows ntfs partition with the help og ntfs-3g that will also enable read and write in that partition. The syntax for this is same as for mounting `ext4` filesystem, just replace paritition filesystem with `ntfs-3g` in the above syntax, here is an example
{% highlight bash linenos %}
UUID=0CC84430C84419FA   /media/data     ntfs-3g     rw      0       0
{% endhighlight %}

## Installing MS Office fonts
MS fonts like ARIAL, Times of Roman etc. can be easily installed using a single command. 
{% highlight bash linenos %}
yay -S ttf-ms-fonts
{% endhighlight %}

## Enabling emoji support
By default linux will not display emoji instead it will display a black square. To enable emoji support follow steps below.
- Install noto fonts emoji
{% highlight bash linenos %}
yay -S noto-fonts-emoji
{% endhighlight %}
- create a config file named emoji.conf inside /etc/fonts/conf.d/
{% highlight bash linenos %}
sudo nano /etc/fonts/conf.d/emoji.conf
{% endhighlight %}
- Edit contents of config file
{% highlight bash linenos %}
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>

    <alias>
        <family>sans-serif</family>
        <prefer>
        <family>Your favorite sans-serif font name</family>
        <family>Noto Color Emoji</family>
        <family>Noto Emoji</family>
        </prefer> 
    </alias>

    <alias>
        <family>serif</family>
        <prefer>
        <family>Your favorite serif font name</family>
        <family>Noto Color Emoji</family>
        <family>Noto Emoji</family>
        </prefer>
    </alias>

    <alias>
    <family>monospace</family>
    <prefer>
        <family>Your favorite monospace font name</family>
        <family>Noto Color Emoji</family>
        <family>Noto Emoji</family>
        </prefer>
    </alias>

</fontconfig>
{% endhighlight %}
If you want to use emoji then installing a emoji picker will help.
- Install emoji picker
{% highlight bash linenos %}
yay -S ibus-uniemoji
{% endhighlight %}
- Create a keyboard shortcut to open emoji picker with shortcut. Use command `ibus emoji` and set your favourite key combination for opening emoji picker. I am using <kbd>Alt + .</kbd> as shortcut.