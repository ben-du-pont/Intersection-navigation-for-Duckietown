# **Intersection Navigation: Final Project ETHZ 2023**

## Group 7 - Duck Squad - Yanni Kechriotis, Samuel Montorfani, Benjamin Dupont 

# About these activities

The Duckietown LX supporting the final projects at ETHZ 2023

In this activity, the navigation of one or more Duckiebots through an intersection is carried out.

Each Duckiebot navigate within a Duckietown following its road. Upon reaching an intersection, it follows the following procedure:

- stop in front of the stop line.
- randomly decides a direction to take based on the type of intersection.
- indicates the chosen direction using its LEDs.
- checks for the presence of other Duckiebots and detects LED signals to understand the directions they intend to take.
- verifies if it has the right of way. If yes, it proceeds, if not, it waits.
- when it can proceed, takes the action to continue in the chosen direction.
- resumes navigating along its road.


# Demo instructions

**Environment setup**
The physical environment for executing this project can be any standard dockietown city with standard intersections. More precisely, the roads should form closed circuits, and the intersections should be marked with red stop lines. Additional components that could be interpreted as red stop lines or other Duckiebots should not be present. Only Duckiebots involved in the simulation should be on the roads.

The maximum number of Duckiebots that can be present is unlimited, provided that each Duckiebot is initially positioned with sufficient distance from the others. This is because the demo does not involve obstacle management outside of an intersection.
The initial position of the Duckiebots must be within a lane but not within an intersection.

You should run the simple.ipynb file, and change the name of your duckiebot in the top cell. 

**expected results**

A Duckiebot should be able to navigate through an intersection correctly, without colliding, and respecting right of way. Additionally, it should stay within the lanes and continue to navigate until the activity is finished.
In the event that a Duckiebot deviates from the lane, it can be manually placed back in the correct position, and the activity should proceed correctly.


# Instructions

**NOTE:** All commands below are intended to be executed from the root directory of this exercise (i.e., the directory containing this README).


## 1. Make sure your exercise is up-to-date

Update your exercise definition and instructions,

    git pull origin merged_branch

**NOTE:** Example instructions to fork a repository and configure to pull from upstream can be found in the [duckietown-lx repository README](https://github.com/duckietown/duckietown-lx/blob/mooc2022/README.md).


## 2. Make sure your system is up-to-date

- 💻 Always make sure your Duckietown Shell is updated to the latest version. See [installation instructions](https://github.com/duckietown/duckietown-shell)

- 💻 Update the shell commands: `dts update`

- 💻 Update your laptop/desktop: `dts desktop update`

- 🚙 Update your Duckiebot: `dts duckiebot update ROBOTNAME` (where `ROBOTNAME` is the name of your Duckiebot chosen during the initialization procedure.)


## 3. Work on this activity

### Launch the code editor

Open the code editor by running the following command,

```
dts code editor
```

Wait for a URL to appear on the terminal, then click on it or copy-paste it in the address bar
of your browser to access the code editor. The first thing you will see in the code editor is
this same document, you can continue there.


### Run the activity

NOTE: You should be reading this from inside the code editor in your browser.

Inside the code editor, use the navigator sidebar on the left-hand side to navigate to the
notebooks directory, then to the project directory, and finally open and run the main.ipynb file.
Additionally you can also play with the simple.ipynb and the simple sam.ipynb files.


## Troubleshooting

If an error of this form occurs

```bash
Traceback (most recent call last):
  File "/usr/local/lib/python3.8/dist-packages/duckietown_challenges_cli/cli.py", line 76, in dt_challenges_cli_main
    dt_challenges_cli_main_(args=args, sections=sections, main_cmd="challenges")
  File "/usr/local/lib/python3.8/dist-packages/duckietown_challenges_cli/cli.py", line 203, in dt_challenges_cli_main_
    f(rest, environment)
  File "/usr/local/lib/python3.8/dist-packages/duckietown_challenges_cli/cli_submit.py", line 165, in dt_challenges_cli_submit
    br = submission_build(
  File "/usr/local/lib/python3.8/dist-packages/duckietown_challenges_cli/cmd_submit_build.py", line 41, in submission_build
    raise ZException(msg, available=list(credentials))
zuper_commons.types.exceptions.ZException: Credentials for registry docker.io not available
available:
```

you need to log into docker using `dts`. Use this command:

```
dts challenges config --docker-username <USERNAME> --docker-password <PASSWORD>
```

