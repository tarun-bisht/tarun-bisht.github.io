FROM ubuntu:22.04
LABEL Maintainer="Tarun Bisht, work@tarunbisht.com"

RUN apt-get update && apt-get -y upgrade
RUN apt-get -y install ruby-full build-essential zlib1g-dev

SHELL ["/bin/bash", "-c"]

RUN echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc && \
    echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc && \
    echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc && \
    source ~/.bashrc

RUN gem install jekyll bundler

EXPOSE 4000
EXPOSE 35729

COPY Gemfile /usr
COPY Gemfile.lock /usr
RUN cd /usr && gem update && bundle install