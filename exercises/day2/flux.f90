subroutine euler_flux(q, flux)
  implicit none
  real :: q(3) ! primitive state [\rho, u, p]
  real :: flux(3) ! euler flux [\rho * u, \rho u^2 + p, (\rho * E + p ) u]
  real, parameter :: R = 287.39
  real, parameter :: gamma_air = 1.4
  real :: T
  real :: rhoH

  ! T = p / \rho * R
  T = q(3) / R * q(1)
  ! \rho H = p * \gamma / (\gamma - 1) + 1/2 \rho u^2
  rhoH = q(3) * gamma_air / (gamma_air - 1) + 0.5 * q(1) * q(2) * q(2)
  ! rho * u
  flux(1) = q(1) * q(2)
  ! \rho u^2 + p
  flux(2) = q(1) * q(2) * q(2) + q(3)
  ! u * \rho H
  flux(3) = q(2) * rhoH
end subroutine euler_flux

