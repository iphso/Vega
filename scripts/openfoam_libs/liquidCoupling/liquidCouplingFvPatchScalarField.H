/*---------------------------------------------------------------------------*\
    liquidCoupling boundary condition (EXPERIMENT_LOG §66)

Description
    A `mixed` (Robin) temperature boundary condition, structurally
    IDENTICAL to OpenFOAM's own built-in `externalWallHeatFluxTemperature`
    convective mode (its updateCoeffs() was read directly out of
    $FOAM_SRC before writing this -- see the .C file for the exact
    correspondence), except the ambient temperature `Ta` is not a fixed
    constant: it is a lumped liquid temperature this BC integrates
    forward itself, one explicit Euler step per solver timestep, from the
    liquid's own energy balance

        m*c*dT/dt = -sum_over_patch( h*(Tliquid - Tface) * faceArea )

    This exists because the alternative -- a runtime-compiled
    `codedMixed` doing the same thing inline in the case dictionary --
    is unconditionally refused by OpenFOAM when running as root
    (dynamicCode::checkSecurity()'s isAdministrator() check has no
    override), which is how this project's `openfoam` Docker service
    runs (matching `train`/`oracle`'s own convention, see Dockerfile.openfoam).
    A real compiled `wmake libso` plugin -- OpenFOAM's own standard
    extension mechanism, the same one real third-party BCs use -- sidesteps
    that restriction entirely AND is compiled once at image-build time
    instead of once per candidate, which also matters for dataset-
    generation throughput (dynamicCode's g++ invocation would otherwise
    cost real wall-clock time on every single candidate).

    Author: none of this is a physics-solver replacement -- OpenFOAM's own
    verified FVM/PIMPLE machinery still does all of the actual conduction
    solve; this is only the same small "how does the liquid lose heat"
    coupling logic any integration approach (Python-side restart loop,
    codedMixed, or this) would need to supply somewhere.

    Dictionary entries:
        h                  heat transfer coefficient, W/(m^2 K)
        mLiquid            liquid mass, kg -- the REAL physical mass (e.g.
                           a full 360-degree mug's worth), not scaled
                           down to whatever slice of it this patch's own
                           mesh covers
        cLiquid            liquid specific heat capacity, J/(kg K)
        Tliquid0           initial liquid temperature, K (only used if no
                           `Tliquid` entry is present, i.e. a fresh start)
        wedgeFraction      (optional, default 1) full_angle/patch_angle --
                           if this patch is only an axisymmetric WEDGE
                           slice of the real body (mug_openfoam_oracle.py's
                           2.5-degree half-angle mesh), `gSum(q*magSf())`
                           over the patch only sees that slice's own heat
                           loss; multiplying by this factor recovers the
                           real total the full body's mLiquid should be
                           losing. A caught bug (§66): the first version
                           of this BC omitted it entirely, so a real mug
                           case's liquid barely cooled (~1/72 of its
                           actual heat loss) while mLiquid was the full
                           physical mass -- silently wrong, not a crash,
                           caught only by comparing temp_at_2h_C against
                           v6 and finding it implausible.
        coatingResistance  (optional, default 0) m^2 K/W, an EXTRA series
                           thermal resistance between the liquid film and
                           the wall's own solid face -- a thin interior
                           glaze/enamel/liner coating (§74), modeled the
                           same way externalWallHeatFluxTemperature's own
                           built-in thicknessLayers/kappaLayers models an
                           exterior coating: added in series with 1/h,
                           not as a separately meshed region (a real
                           glaze is far too thin, ~0.1-0.5mm, to mesh
                           affordably at dataset-generation scale).
                           thickness/kappa, precomputed on the Python
                           side rather than passed as two separate
                           entries -- only the ratio matters physically.

\*---------------------------------------------------------------------------*/

#ifndef liquidCouplingFvPatchScalarField_H
#define liquidCouplingFvPatchScalarField_H

#include "mixedFvPatchFields.H"

namespace Foam
{

class liquidCouplingFvPatchScalarField
:
    public mixedFvPatchScalarField
{
    // Private Data

        scalar h_;
        scalar mLiquid_;
        scalar cLiquid_;
        scalar wedgeFraction_;
        scalar coatingResistance_;

        //- Current lumped liquid temperature (K) -- evolves over the run
        scalar Tliquid_;

        //- Simulated time (s) the liquid ODE was last advanced at, so
        //  updateCoeffs() (which chtMultiRegionFoam's outer coupling loop
        //  may call more than once per timestep) only integrates once
        //  per real timestep
        scalar lastUpdateTime_;


public:

    //- Runtime type information
    TypeName("liquidCoupling");


    // Constructors

        liquidCouplingFvPatchScalarField
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&
        );

        liquidCouplingFvPatchScalarField
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const dictionary&
        );

        liquidCouplingFvPatchScalarField
        (
            const liquidCouplingFvPatchScalarField&,
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const fvPatchFieldMapper&
        );

        liquidCouplingFvPatchScalarField
        (
            const liquidCouplingFvPatchScalarField&
        ) = delete;

        liquidCouplingFvPatchScalarField
        (
            const liquidCouplingFvPatchScalarField&,
            const DimensionedField<scalar, volMesh>&
        );

        virtual tmp<fvPatchScalarField> clone
        (
            const DimensionedField<scalar, volMesh>& iF
        ) const
        {
            return tmp<fvPatchScalarField>
            (
                new liquidCouplingFvPatchScalarField(*this, iF)
            );
        }


    // Member Functions

        //- Current lumped liquid temperature (K) -- the oracle reads
        //  this back out after the run to report temp_at_2h_C
        scalar Tliquid() const
        {
            return Tliquid_;
        }

        virtual void updateCoeffs();

        virtual void write(Ostream&) const;
};

} // End namespace Foam

#endif
