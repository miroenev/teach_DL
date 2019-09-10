https://stackoverflow.com/questions/40195740/how-to-run-openai-gym-render-over-a-server/45179251#45179251


1 - build docker container image
	* requires that you create an account on compute.nvidia.com
	* building blocks include : pytorch, jupyter, openAI gym

2 - launch container using the shell script [ launch_container_with_x11_tunnel.sh ]
	* note this script may need to be made executable [ e.g. $ chmod +x launch_container_with_x11_tunnel.sh ]
	* the container is launched with Xauthority credentials copied from the host to enable X11 tunneling

3 - inside container launch jupyter with a virtual framebuffer for rendering world/game state	
	* $ xvfb-run -s "-screen 0 800x600x24" jupyter notebook --allow-root --no-browser --ip 0.0.0.0 

98fdf64cf9014ebdd2966fb15118cb6c692b85b15f7ebe26
