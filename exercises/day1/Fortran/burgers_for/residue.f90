! Copy paste the differentiated subroutine from Tapenade
SUBROUTINE RESIDUE_DA(nu, delta, a, ad, res, resd)
  IMPLICIT NONE
  REAL*8, INTENT(IN) :: nu, delta, a
  REAL*8, INTENT(IN) :: ad
  REAL*8, INTENT(OUT) :: res
  REAL*8, INTENT(OUT) :: resd
  write(*, *) "Fix me with the right gradient in residue.f90"
  STOP
END SUBROUTINE RESIDUE_DA

! residue function
subroutine residue (nu, delta, A, res)
  implicit none
  real(8), intent(in)  :: nu, delta, A
  real(8), intent(out) :: res
  res = ( 1.00d0 + delta + A * A ) * tanh(A / nu) - (2.00d0 + delta) * A
end subroutine residue

! objective function
subroutine objective (nu, A, x0)
  implicit none
  real(8), intent(in)  :: nu, A
  real(8), intent(out) :: x0
  x0 = 1.00d0 - 2.00d0 * nu / A * atanh(1.00d0 / A)
end subroutine objective

! steady-state solution
subroutine solution (nu, A, x0, x, u )
  implicit none
  real(8), intent(in)  :: nu, A, x0, x
  real(8), intent(out) :: u
  u = -A * tanh( 0.50d0 * A / nu * ( x - x0 ) )
end subroutine solution

! root finding algorithm
subroutine find_root( nu, delta, A, eps, iter_max  )
  implicit none
  real(8), intent(in)  :: nu, delta, eps
  integer, intent(in)  :: iter_max
  real(8), intent(out) :: A
  real(8) :: fA, fAd, update
  real(8), parameter :: Ad = 1.00d0
  integer :: i

  do i = 1, iter_max
    fAd = 0.00d0
    call residue_dA(nu, delta, A, Ad, fA, fAd)
    update = fA / fAd
    if ( abs(update) .lt. eps ) return
    A = A - fA / fAd
  end do

end subroutine find_root

