module chpes_lib
    use iso_c_binding
    implicit none
    public :: chpes
    interface
      function create_chpes_c(chcell, chpbc, chnumatoms, chspecies, chmaxnumtype, chatomtype, chmass) bind(C, name="create_chpes")
        use iso_c_binding
        implicit none
        type(c_ptr) :: create_chpes_c
        integer(c_int), value :: chnumatoms
        integer(c_int) :: chpbc(3)
        real(c_double), dimension(3,3) :: chcell
        type(c_ptr), dimension(chnumatoms)::chspecies 
        integer(c_int), value :: chmaxnumtype
        type(c_ptr), dimension(chmaxnumtype)::chatomtype
        real(c_double), dimension(chnumatoms) :: chmass
      end function
    
      subroutine delete_chpes_c(chpes) bind(C, name="delete_chpes")
        use iso_c_binding
        implicit none
        type(c_ptr), value :: chpes
      end subroutine
    
      subroutine pes_reann_chout_c(chpes, chnumatoms, chcart, chenergy, chforce) bind(C, name="pes_reann_chout")
        use iso_c_binding
        implicit none
        type(c_ptr), intent(in), value :: chpes
        integer(c_int), value :: chnumatoms
        real(c_double), dimension(chnumatoms*3), intent(in) :: chcart
        real(c_double), dimension(chnumatoms),intent(inout) :: chenergy
        real(c_double), dimension(chnumatoms*chnumatoms*3), intent(inout) :: chforce
      end subroutine
    end interface

    type chpes
      private
      type(c_ptr) :: chptr
      integer::chnumatoms
    contains
      final :: delete_chpes
      procedure :: chdelete => delete_chpes_polymorph
      procedure :: reann_chout => pes_reann_chout
    end type

    interface chpes
      procedure create_chpes
    end interface

contains 
    function create_chpes(chcell, chpbc, chnumatoms, chspecies_f, chmaxnumtype, chatomtype_f, chmass)
      implicit none
      type(chpes) :: create_chpes
      double precision, dimension(3,3),intent(in) :: chcell
      integer, dimension(3),intent(in):: chpbc(3)
      integer,intent(in) :: chnumatoms
      character(len=2, kind=C_CHAR), dimension(chnumatoms),intent(in) :: chspecies_f
      character(len=3, kind=C_CHAR), dimension(chnumatoms), target:: chspecies_c
      type(c_ptr), dimension(chnumatoms)::chspecies
      integer,intent(in) :: chmaxnumtype
      character(len=2, kind=C_CHAR), dimension(chmaxnumtype),intent(in) :: chatomtype_f
      character(len=3, kind=C_CHAR), dimension(chmaxnumtype), target:: chatomtype_c
      type(c_ptr), dimension(chmaxnumtype)::chatomtype
      double precision, dimension(chnumatoms),intent(in) :: chmass
      integer :: i
      do i=1,chnumatoms
        chspecies_c(i)= chspecies_f(i)//C_NULL_CHAR
        chspecies(i)= c_loc(chspecies_c(i))
      end do
      do i=1,chmaxnumtype
        chatomtype_c(i)= chatomtype_f(i)//C_NULL_CHAR
        chatomtype(i)= c_loc(chatomtype_c(i))
      end do
      create_chpes%chptr = create_chpes_c(chcell,chpbc,chnumatoms, chspecies,chmaxnumtype,chatomtype, chmass)
      create_chpes%chnumatoms = chnumatoms
    end function

    subroutine delete_chpes(this)
      implicit none
      type(chpes) :: this
      call delete_chpes_c(this%chptr)
    end subroutine

    subroutine delete_chpes_polymorph(this)
      implicit none
      class(chpes) :: this
      call delete_chpes_c(this%chptr)
    end subroutine

    subroutine pes_reann_chout(this, chcart, chenergy, chforce)
      implicit none
      class(chpes), intent(in) :: this
      double precision, dimension(this%chnumatoms*3), intent(in) :: chcart
      double precision, dimension(this%chnumatoms),intent(inout) :: chenergy
      double precision, dimension(this%chnumatoms*this%chnumatoms*3), intent(inout) :: chforce
      call pes_reann_chout_c(this%chptr,this%chnumatoms, chcart, chenergy, chforce)
    end subroutine

end module

module chpes_mod
  use chpes_lib
  implicit none
  type(chpes) :: chpes_ptr
  real(kind=8)::chcell(3,3)
  integer(kind=4)::chpbc(3)
  integer(kind=4)::chnumatoms,chmaxnumtype
  character(len=2),allocatable::chspecies(:),chatomtype(:)
  real(kind=8),allocatable::chmass(:)
end module

subroutine chreadinput()
  use chpes_mod
  implicit none
  integer(kind=4) i
  open(428,file="./input_reann")
  read(428,*)
  do i = 1,3
    read(428,*) chcell(i,1:3)
  end do
  read(428,*)
  read(428,*) chpbc
  read(428,*)
  read(428,*) chnumatoms
  read(428,*)
  allocate(chspecies(chnumatoms),chmass(chnumatoms))
  do i=1,chnumatoms
    read(428,*) chspecies(i),chmass(i)
  enddo
  read(428,*)
  read(428,*) chmaxnumtype
  allocate(chatomtype(chmaxnumtype))
  read(428,*)
  do i=1,chmaxnumtype
    read(428,*) chatomtype(i)
  enddo      
  close(428)
end subroutine

subroutine init_chreann()
  use chpes_mod
  implicit none
  call chreadinput()
  chpes_ptr=chpes(chcell,chpbc,chnumatoms, chspecies, chmaxnumtype, chatomtype, chmass)
end subroutine

subroutine delete_chreann()
  use chpes_mod
  implicit none
  deallocate(chspecies,chmass,chatomtype)
  call chpes_ptr%chdelete
end subroutine 
