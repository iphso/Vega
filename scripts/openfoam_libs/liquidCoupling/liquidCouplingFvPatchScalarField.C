#include "liquidCouplingFvPatchScalarField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "solidThermo.H"
#include "physicalProperties.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::liquidCouplingFvPatchScalarField::liquidCouplingFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF
)
:
    mixedFvPatchScalarField(p, iF),
    h_(0),
    mLiquid_(1),
    cLiquid_(4186),
    wedgeFraction_(1),
    coatingResistance_(0),
    Tliquid_(293.15),
    lastUpdateTime_(-1)
{
    refValue() = 0;
    refGrad() = 0;
    valueFraction() = 1;
}


Foam::liquidCouplingFvPatchScalarField::liquidCouplingFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict
)
:
    mixedFvPatchScalarField(p, iF),
    h_(dict.lookup<scalar>("h")),
    mLiquid_(dict.lookup<scalar>("mLiquid")),
    cLiquid_(dict.lookup<scalar>("cLiquid")),
    wedgeFraction_(dict.lookupOrDefault<scalar>("wedgeFraction", 1.0)),
    coatingResistance_(dict.lookupOrDefault<scalar>("coatingResistance", 0.0)),
    Tliquid_
    (
        dict.found("Tliquid")
      ? dict.lookup<scalar>("Tliquid")
      : dict.lookup<scalar>("Tliquid0")
    ),
    lastUpdateTime_(dict.lookupOrDefault<scalar>("lastUpdateTime", -1))
{
    fvPatchScalarField::operator=(scalarField("value", dict, p.size()));

    if (dict.found("refValue"))
    {
        refValue() = scalarField("refValue", dict, p.size());
        refGrad() = scalarField("refGradient", dict, p.size());
        valueFraction() = scalarField("valueFraction", dict, p.size());
    }
    else
    {
        refValue() = *this;
        refGrad() = 0;
        valueFraction() = 1;
    }
}


Foam::liquidCouplingFvPatchScalarField::liquidCouplingFvPatchScalarField
(
    const liquidCouplingFvPatchScalarField& ptf,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    mixedFvPatchScalarField(ptf, p, iF, mapper),
    h_(ptf.h_),
    mLiquid_(ptf.mLiquid_),
    cLiquid_(ptf.cLiquid_),
    wedgeFraction_(ptf.wedgeFraction_),
    coatingResistance_(ptf.coatingResistance_),
    Tliquid_(ptf.Tliquid_),
    lastUpdateTime_(ptf.lastUpdateTime_)
{}


Foam::liquidCouplingFvPatchScalarField::liquidCouplingFvPatchScalarField
(
    const liquidCouplingFvPatchScalarField& ptf,
    const DimensionedField<scalar, volMesh>& iF
)
:
    mixedFvPatchScalarField(ptf, iF),
    h_(ptf.h_),
    mLiquid_(ptf.mLiquid_),
    cLiquid_(ptf.cLiquid_),
    wedgeFraction_(ptf.wedgeFraction_),
    coatingResistance_(ptf.coatingResistance_),
    Tliquid_(ptf.Tliquid_),
    lastUpdateTime_(ptf.lastUpdateTime_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::liquidCouplingFvPatchScalarField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    const fvPatch& p = this->patch();
    const fvMesh& mesh = p.boundaryMesh().mesh();
    const scalarField& Tp = *this;

    const scalar t = mesh.time().value();
    const scalar dt = mesh.time().deltaTValue();

    // Same lookup temperatureCoupledBase::kappa() uses for a
    // solidThermo-backed patch (confirmed by reading it directly out of
    // $FOAM_SRC -- this project's own already-validated
    // externalWallHeatFluxTemperature BC on the `ambient` patch goes
    // through the exact same call).
    const solidThermo& thermo =
        mesh.lookupObject<solidThermo>(physicalProperties::typeName);
    const scalarField kappaw(thermo.kappa().boundaryField()[p.index()]);

    // §74: a coating (glaze/enamel/liner) between the liquid film and the
    // solid wall is an EXTRA series resistance, same algebra
    // externalWallHeatFluxTemperature's own totalSolidRes uses for an
    // exterior coating -- combine it with the film coefficient into one
    // effective h before doing anything else with h_ below.
    const scalar hEff = 1.0/(1.0/h_ + coatingResistance_);

    // Advance the lumped liquid ODE once per REAL timestep (not once per
    // updateCoeffs() call -- chtMultiRegionFoam's outer region-coupling
    // loop can call this more than once per timestep).
    if (t > lastUpdateTime_ + 0.5*dt)
    {
        // Tp here is still last timestep's converged boundary-face value
        // (this call happens before this timestep's solve) -- an
        // explicit one-step-lagged coupling, same convention
        // reference_1d_coupled.py independently implements for its own
        // cross-check.
        const scalarField q(hEff*(Tliquid_ - Tp));
        // gSum over just THIS patch's own faces -- if it's a wedge slice
        // of a full axisymmetric body (wedgeFraction_ > 1), scale up to
        // the real total the full-mass liquid is actually losing.
        const scalar Qtotal = gSum(q*p.magSf()) * wedgeFraction_;
        Tliquid_ -= dt*Qtotal/(mLiquid_*cLiquid_);
        lastUpdateTime_ = t;
    }

    // Same Robin-BC algebra as externalWallHeatFluxTemperature's
    // convective branch with no radiation/extra flux term (qTot=0):
    // refValue=Ta, valueFraction=hEff/(hEff+kappa*deltaCoeffs).
    const scalarField kappaDeltaCoeffs(kappaw*p.deltaCoeffs());
    refValue() = Tliquid_;
    refGrad() = 0;
    valueFraction() = hEff/(hEff + kappaDeltaCoeffs);

    mixedFvPatchScalarField::updateCoeffs();
}


void Foam::liquidCouplingFvPatchScalarField::write(Ostream& os) const
{
    fvPatchScalarField::write(os);
    writeEntry(os, "h", h_);
    writeEntry(os, "mLiquid", mLiquid_);
    writeEntry(os, "cLiquid", cLiquid_);
    writeEntry(os, "wedgeFraction", wedgeFraction_);
    writeEntry(os, "coatingResistance", coatingResistance_);
    // Current evolved state, not the original "Tliquid0" input -- a
    // restart from a written time directory should resume from where
    // the liquid actually was, not reset back to the initial condition.
    writeEntry(os, "Tliquid", Tliquid_);
    writeEntry(os, "lastUpdateTime", lastUpdateTime_);
    writeEntry(os, "refValue", refValue());
    writeEntry(os, "refGradient", refGrad());
    writeEntry(os, "valueFraction", valueFraction());
    writeEntry(os, "value", *this);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchScalarField,
        liquidCouplingFvPatchScalarField
    );
}

// ************************************************************************* //
