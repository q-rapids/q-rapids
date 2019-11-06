# Q-Rapids Contribution Guidelines

The OpenReq project is a research project funded by the European Union Horizon 2020 Research and Innovation programme under grant agreement No 732253.

There are several ways to contribute to Q-Rapids project

1. Give us your feedback
2. Testing
3. Contributing with Code
4. Documenting

# Give us your feedback

We are interested in your opinion about the Q-Rapids tool. You can submit an issue report in the corresponding GitHub repository or use the contact page in the [Q-Rapids website](http://www.q-rapids.eu/contact).

You can find some details for submitting an issue report in the [last section of this file](https://github.com/q-rapids/q-rapids/blob/master/CONTRIBUTING.md#how-to-submit-an-issue-report).

# Testing

Test new Q-Rapids tool versions and report bugs opening issues in the corresponding GitHub repository. It is important to report bugs correctly giving the sufficient detail to reproduce them.

You can find some details for submitting an issue report in the [last section of this file](https://github.com/q-rapids/q-rapids/blob/master/CONTRIBUTING.md#how-to-submit-an-issue-report).


# Contributing with Code

In order to separate the contributions from the different developers groups, there are several repositories in the GitHub where developers can contribute.

Before submitting new code
- Please make sure that you’re respecting the license of the corresponding component, adding the corresponding headers to the new files.
- Always create a new feature branch for your code contribution based on the master branch.
- If your contribution is realted to a known issues, please include a reference to the issue in the commit comment.

You can find some development guidelines in this [repository's Wiki](https://github.com/q-rapids/q-rapids/wiki/Contributing-with-code)

# Documenting
If you think that something can be improved, open a issue in the corresponding GitHub repository.

If you consider that something is missing, you can also add new pages or add some direct links to the sidebar locate at the right.

<!--- If you need to include some images in the wiki pages, we need to store them in the folder "images" for this wiki. For clonning the wiki repository, you can execute:

` git clone git@github.com:riscoss/riscoss-platform-core.wiki.git`

This repository contains a folder named `images`, where you can add the images you need. 

For using these images in the Wiki pages, you need to use a sentence like:

`[[wiki/images/logo_riscoss_DSP.png]]` for references in the wiki pages <br>

or <br>

`![](https://github.com/RISCOSS/riscoss-platform-core/wiki/images/logo_riscoss_DSP.png)` for references you need the full path (e.g. the readme.md file)
--->



# How to submit an issue report

Before submitting a new issue
- Please make sure that you’re using the latest version of the component. 
- It would be very much appreciated if you would search the project bug tracker to avoid creating a duplicate issue.

Your issue should include at least the following information:
- If you report a bug, please include steps to reproduce it. Please add a step by step documentation and screenshots if applicable and your operating system.
- If you report an enhancement request, please describe enough details so that others can understand it.

This project uses a set of labels that are used to differentiate 3 kinds of issues:
1. **"bug"**: bugs/malfunctions.
2. **"enhancement"**: new features or improvements.
3. **"question"** or **"help wanted"**: asking for assistance.

It is recommended to use always one of these 3 labels to characterise the issue.

Besides these labels, there are 2 labels to indicate that the issue is special:
1. **"blocking"**: a critical issue, something to be handled as soon as possible.
2. **"nicetohave"**: a low priority issue. 

At some point, a (non-help for assistance) issue can be closed without include any modification in the code, in this case the issue will be labelled using:
1. **"duplicate"**: there are more than one issue concerning the same fact. In this case, the person closing the issue will include the reference of the issue that is kept as a comment (e.g. #13).
2. **"invalid"** or **"wontfix"** if the issue is discarded. In this case, the person closing the issue will include the reason as a comment.

