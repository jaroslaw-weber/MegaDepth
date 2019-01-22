FROM pytorch/pytorch

# model download
RUN wget https://www.cs.cornell.edu/projects/megadepth/dataset/models/best_generalization_net_G.pth
RUN cp best_generalization_net_G.pth /workspace/checkpoints/test_local/best_vanila_net_G.pth

# install dependencies 
RUN apt-get update \
	&& apt-get install -y \
    wget \
	imagemagick \
	micro

#install python dependencies
RUN pip install scipy
RUN pip install scikit-image

#create folders
RUN mkdir checkpoints
RUN mkdir checkpoints/test_local
RUN pip install git+https://github.com/jaroslaw-weber/depthgen