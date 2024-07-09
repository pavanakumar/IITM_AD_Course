  !$acc routine seq
  subroutine Relu(n, xin, xout) bind(C)
    use iso_c_binding
    implicit none
    integer(c_int32_t), intent(in) :: n
    real(c_float), intent(in)    :: xin(n)
    real(c_float), intent(out)   :: xout(n)
    logical :: mask(n)
    xout = xin
    mask = xin .le. 0
    where(mask)
       xout = 0
    end where
  end subroutine Relu

  !$acc routine seq
  subroutine LeakyRelu(n, xin, xout) bind(C)
    use iso_c_binding
    implicit none
    integer(c_int32_t), intent(in) :: n
    real(c_float), intent(in)    :: xin(n)
    real(c_float), intent(out)   :: xout(n)
    logical :: mask(n)
    xout = xin
    mask = xin .le. 0
    where(mask)
       xout = 0.1d0 * xin
    end where
  end subroutine LeakyRelu

  !$acc routine seq
  subroutine TangentHyperbolic(n, xin, xout) bind(C)
    use iso_c_binding
    implicit none
    integer(c_int32_t), intent(in) :: n
    real(c_float), intent(in)    :: xin(n)
    real(c_float), intent(out)   :: xout(n)
    xout = tanh(xin)
  end subroutine TangentHyperbolic

  !$acc routine seq
  subroutine identity(n, xin, xout) bind(C)
    use iso_c_binding
    implicit none
    integer(c_int32_t), intent(in) :: n
    real(c_float), intent(in)    :: xin(n)
    real(c_float), intent(out)   :: xout(n)
      xout = xin
  end subroutine identity

  !$acc routine seq
  subroutine LayerNorm(n, scal, ofs, xin, xout) bind(C)
    use iso_c_binding
    implicit none
    integer(c_int32_t), intent(in) :: n
    real(c_float), intent(in)    :: scal(n)
    real(c_float), intent(in)    :: ofs(n)
    real(c_float), intent(in)    :: xin(n)
    real(c_float), intent(out)   :: xout(n)
    real(c_float), parameter     :: eps = 1.0d-5
    real(c_float) :: mean
    real(c_float) :: var
    xout(1:n) = 0
    mean = sum(xin(1:n)) / n
    var  = sum((xin(1:n) - mean)**2) / n
    xout(1:n) = (xin(1:n) - mean) / (sqrt(var) + eps)
    xout(1:n) = xout(1:n) * scal(1:n) + ofs(1:n)
  end subroutine LayerNorm

  !$acc routine seq
  subroutine DenseLayer(nin, nout, W, b, xin, xout) bind(C)
    use iso_c_binding
    implicit none
    integer(c_int32_t), intent(in) :: nin
    integer(c_int32_t), intent(in) :: nout
    real(c_float), intent(in)    :: W(nin, nout)
    real(c_float), intent(in)    :: b(nout)
    real(c_float), intent(in)    :: xin(nin)
    real(c_float), intent(out)   :: xout(nout)
    integer(c_int32_t) :: i, j
    xout(1:nout) = 0
    !$AD II-LOOP acc loop seq
    do j = 1, nout
      do i = 1, nin
        xout(j) = xout(j) + W(i, j) * xin(i)
      end do
    end do
    xout = xout + b
  end subroutine DenseLayer

subroutine Encoder_2(& 
  &  nnodes,  &
  &  nedges,  &
  &  edgelist,  &
  &  params,  &
  &  x,  &
  &  xout3) &
  &  bind(C, name="Encoder_2")
  use iso_c_binding
  implicit none
  integer(c_int32_t) , intent(in) :: nnodes
  integer(c_int32_t) , intent(in) :: nedges
  integer(c_int32_t), dimension(2,nedges) , intent(in) :: edgelist
  real(c_float), dimension(461) , intent(in) :: params
  real(c_float), dimension(1,nnodes) , intent(in) :: x
  real(c_float), dimension(1,nnodes) , intent(out) :: xout3
  real(c_float) :: xin(1)
  real(c_float) :: x1(10)
  real(c_float) :: x1_noact(10)
  real(c_float) :: x1_act(10)
  real(c_float) :: x2(20)
  real(c_float) :: x2_noact(20)
  real(c_float) :: x2_act(20)
  real(c_float) :: x3(10)
  real(c_float) :: x3_noact(10)
  real(c_float) :: x3_act(10)
  real(c_float) :: xout(1)
  real(c_float) :: xout_noact(1)
  real(c_float) :: xout_act(1)

  integer(c_int32_t) :: i, i1, i2, iedge, inode
  ! Zero-out output array
  !$AD II-LOOP acc parallel loop collapse(2) present(xout3)
  do inode = 1, nnodes
    do i = 1, 1
        xout3(i, inode)= 0
    end do
  end do
  !$AD II-LOOP acc parallel loop gang vector present(edgelist, params, x, xout3) private(xin, x1, x2, x3, xout, x1_noact, x2_noact, x3_noact, xout_noact, x1_act, x2_act, x3_act, xout_act)
  do inode = 1, nnodes
    ! Copy contents into temporary for MLP
    !$AD II-LOOP acc loop seq
    do i = 1, 1
      xin(i + 0) = x(i, inode)
    end do
    ! Layer 1
    call denselayer(1, 10, params(1), params(11), xin, x1_noact)
    call tangenthyperbolic(10, x1_noact, x1_act)
    !$AD II-LOOP acc loop seq
    do i = 1, 10
        x1(i) = x1_act(i)
    end do
    ! Layer 2
    call denselayer(10, 20, params(21), params(221), x1, x2_noact)
    call tangenthyperbolic(20, x2_noact, x2_act)
    !$AD II-LOOP acc loop seq
    do i = 1, 20
        x2(i) = x2_act(i)
    end do
    ! Layer 3
    call denselayer(20, 10, params(241), params(441), x2, x3_noact)
    call tangenthyperbolic(10, x3_noact, x3_act)
    !$AD II-LOOP acc loop seq
    do i = 1, 10
        x3(i) = x3_act(i)
    end do
    ! Layer 4
    call denselayer(10, 1, params(451), params(461), x3, xout_noact)
    call identity(1, xout_noact, xout_act)
    !$AD II-LOOP acc loop seq
    do i = 1, 1
        xout(i) = xout_act(i)
    end do
    ! Copy contents from temporary of MLP into output feature
    !$AD II-LOOP acc loop seq
    do i = 1, 1
      xout3(i, inode) = xout(i)
    end do
  end do
end subroutine Encoder_2

subroutine run_network(& 
  &  nnodes,  &
  &  nedges,  &
  &  edgelist,  &
  &  params,  &
  &  EdgeFeature1,  &
  &  x,  &
  &  xout3) &
  &  bind(C, name="run_network")
  use iso_c_binding
  implicit none
  integer(c_int32_t) , intent(in) :: nnodes
  integer(c_int32_t) , intent(in) :: nedges
  integer(c_int32_t), dimension(2,nedges) , intent(in) :: edgelist
  real(c_float), dimension(461) , intent(in) :: params
  real(c_float), dimension(1,nedges) , intent(in) :: EdgeFeature1
  real(c_float), dimension(1,nnodes) , intent(in) :: x
  real(c_float), dimension(1,nnodes) , intent(out) :: xout3

  !$acc data copyin(edgelist(1:2, 1:nedges), params(1:461), EdgeFeature1(1:1, 1:nedges), x(1:1, 1:nnodes)) copyout(xout3(1:1, 1:nnodes))
  call Encoder_2( &
 &  nnodes, &
 &  nedges, &
 &  edgelist, &
 &  params(1:461), &
 &  x, &
 &  xout3 )
!$acc end data
end subroutine run_network
