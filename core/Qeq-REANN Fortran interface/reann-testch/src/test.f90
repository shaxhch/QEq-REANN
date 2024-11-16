program mypes 
    use chpes_mod
    implicit none
    character*40 a
    integer(kind=4) :: i,j,npoint,istat
    real(kind=8),allocatable :: energy_cal(:)
    real(kind=8),allocatable::coor(:), force_ini(:), force_cal(:)

    !part-1 reann model initialization
    call init_chreann()

    allocate(energy_cal(chnumatoms))
    allocate(coor(chnumatoms*3),force_ini(chnumatoms*3),force_cal(chnumatoms*chnumatoms*3))
    npoint=0
    open(181,file="configuration")
    open(182,file="reann")

    do
      read(181,*,iostat=istat)
      do j = 1,3
        read(181,*,iostat=istat) 
      end do
      read(181,*,iostat=istat)
      do j=1,chnumatoms
        read(181,*,iostat=istat)
      end do
      read(181,*,iostat=istat)
      if (istat/=0) exit
      npoint=npoint+1
    end do
    rewind(181)
!    write(*,*) npoint  
    do i = 1,npoint
      read(181,*)
      do j = 1,3
        read(181,*) chcell(j,1:3)
      end do
      read(181,*) a,chpbc
      do j=1,chnumatoms
        read(181,*) chspecies(j),chmass(j),coor(1+(j-1)*3:3+(j-1)*3)
      end do
      read(181,*) a
      energy_cal=0.0
      force_cal=0.0

      !part-2 reann model inference
      call Chpes_ptr%reann_chout(coor, energy_cal, force_cal)

      write(182,'(A8,I5,26F14.8)') 'Point=',i,energy_cal(:)
      do j=1,chnumatoms
        write(182,'(3F12.6,3F14.8)') coor(1+(j-1)*3:3+(j-1)*3)
      end do
    end do
    close(181);close(182)

    !part-3 deallocate all variables realeated to reann
    call delete_Chreann()
    deallocate(coor,force_ini,force_cal)
end program
