QEqLR is a reformulation of charge equilibration (QEq) method of Rappe and Goddard but include lattice effects
in the interaction energy between atom via Ewald summation. Also, it shields the real space interaction 
of atoms at close distance in the central unit cell.
The code QEqLR.py takes two input file: dumpfile of atomic trajectory from LAMMPS' md simulations and the QEq parameter file (param.qeq).
The code runs to output atom ids and their partial charges.
