subroutine stencil(n, coeff, u, res)
  implicit none
  integer :: n
  real :: coeff(3)
  real :: u(n)
  real :: res(n)
  integer :: i

  ! BC left
  res(1) = coeff(2) * u(1) + coeff(3) * u(2)

  ! BC right
  res(n) = coeff(1) * u(n - 1) + coeff(2) * u(n)

  do i = 2, n - 1
    res(i) = coeff(1) * u(i - 1) + coeff(2) * u(i) + coeff(3) * u(i + 1)
  end do

end subroutine stencil

