!        Generated by TAPENADE     (INRIA, Ecuador team)
!  Tapenade 3.16 (bugfix_servletAD) -  4 Jan 2024 17:44
!
!  Differentiation of foo in reverse (adjoint) mode:
!   gradient     of useful results: y
!   with respect to varying inputs: x y
!   RW status of diff variables: x:out y:in-out
SUBROUTINE FOO_B(n, x, xb, y, yb)
  IMPLICIT NONE
  INTEGER :: n
  REAL :: x(n)
  REAL :: xb(n)
  REAL :: y(n)
  REAL :: yb(n)
  INTEGER :: i, iloc_array(1), iloc
  INTRINSIC MAXLOC
  REAL :: tempb
  iloc_array = MAXLOC(x)
  iloc = iloc_array(1)
  xb = 0.0
!$BWD-OF II-LOOP 
  DO i=1,n
    tempb = yb(i)/x(iloc)
    yb(i) = 0.0
    xb(i) = xb(i) + tempb
    xb(iloc) = xb(iloc) - x(i)*tempb/x(iloc)
  END DO
END SUBROUTINE FOO_B

