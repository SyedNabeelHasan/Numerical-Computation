
# Diffusion Dynamics 

Diffusion dynamics is a python based finite difference solver package/repo that allows a user to solve complex physical systems based on advnaced physics & mathematics.  

Currently the repository offers:
> Gauss-Siedel solver : 
        The Gauss-Siedel or GS solver is a python written code that allows the user to solve huge sets of linear system of equations that are sparse in nature with tolerence limit in control of user. 

> Boundary Value type Differntial equation solver :
        This helps the user to use an implicit finite difference  scheme to solve one dimensional Ordinary Differential Equations and visualise the result in form of graph.
        
> T2dHC solver:
        The solver is based on finite difference and follows implicit scheme to solve the problem of heat conduction and thus computing out the temperature field. This is a transient temperature field solver based upon the laws of conduction of heat. 
        The solver is also capable of solving problem of heat-conduction even when there is a boundary condition inside the domain which is continously changing it's location. The user must know a basic python to utilize the complete potential of the solver so that user can define the path of boundary-condition inside the geometry. 
        There is a video below which shows a complex problem of heat conduction solved to compute the transient temperature field.
    

> Pressure correction poisson equation model 3:
        The solver is also based on finite difference and follows semi-implicit scheme to solve problems of incompressible fluid flows and thus compute pressure fields and velocity fields. The user must note that since solver is not fully implicit hence it is conditionally-stable.
        The is Currently the solver utilzes matplolib plots to visualise the results; but sooner we will upload systems to obtain more complex flow field visualisation on advnaced post-processing softwares like paraview. The solver has been tried and tested for the problem of lid-driven cavity at Re = 100 and the results of the solver almost match the standard results of Ghia-et-al results for Re = 100.
>

T2dhC solver result video:
https://github.com/user-attachments/assets/8d7f599f-9086-443e-8ea6-02892db52980


Results of Pressure correction poisson equation model 3: 
![Figure_1](https://github.com/user-attachments/assets/53e44488-4d45-4175-884d-8d9098de4aed)
![Screenshot 2025-04-03 005621](https://github.com/user-attachments/assets/679a0e77-5bf4-48bb-ac1c-daeb7a71e552)
The upcoming solver to support visualisation on ParaView:
![Screenshot 2025-04-17 221359](https://github.com/user-attachments/assets/48c7bbae-87e3-4b6d-ad4d-b1a0cfc52b3d)

