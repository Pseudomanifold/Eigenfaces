# Eigenfaces

A simple script for performing eigenface analysis. Head over to [this
blog
post](http://bastian.rieck.ru/blog/posts/2015/eigenfaces_reconstruction)
for more details.

# Usage

Use

    $ python eigenfaces.py Data/Yale\ faces/Original 100

to perform a reconstruction using 100 randomly-selected faces with
eigensystem reductions of 95% explained variance. If you want to modify
the explained variance, use the optional `--variance` switch:

    $ python eigenfaces.py Data/Yale\ faces/Original 100 --variance 0.99

This results in an eigensystem that explains at least 99% of the
variance. The script will display the reconstruction&nbsp;(i.e. the
projection onto &ldquo;face space&rdquo;) on the left-hand side and the
original image on the right-hand side. Better results are generally
obtained using the `Cropped` subdirectory:

    $ python eigenfaces.py Data/Yale\ faces/Cropped 100

# Data

If you want even more data sets, here are some interesting ones:

- [*The Database of Faces*, AT&T Laboratories Cambridge](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)
- [*The normalized Yale face database*](http://vismod.media.mit.edu/vismod/classes/mas622-00/datasets)
