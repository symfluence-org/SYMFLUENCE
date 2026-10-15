! SYMFLUENCE experimental finite-ice guard, serial MESH CLASS 3.6.
! Units: kg/m2 of native glacier tile; flux columns are integrated per step.
! Fixed thermal geometry: explicitly stop BEFORE committing an exhausting step.
module glacier_ledger
 use, intrinsic :: iso_fortran_env, only: real64
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 implicit none
 integer, allocatable :: mids(:)
 logical, allocatable :: ice(:)
 real(real64), allocatable :: mass(:), startmass(:), flux(:,:), cumulative(:,:)
 logical :: finite=.false., active=.false.
 integer :: step=0, logunit=0
 character(1024) :: pending_restart=''
 integer :: pending_stamp(4)=0
 logical :: has_pending_stamp=.false.
contains
 subroutine ledger_init(ids,sand,lo,hi)
  integer,intent(in)::ids(:),lo,hi
  real,intent(in)::sand(:)
  integer::n,u,ios,k,j,id,rows
  real(real64)::amount
  logical::exists
  logical,allocatable::seen(:)
  n=size(ids)
  allocate(mids(n),ice(n),mass(n),startmass(n),flux(n,5),cumulative(n,5),seen(n))
  mids=ids; ice=nint(sand)==-4; active=any(ice); mass=0; flux=0; cumulative=0; seen=.false.
  if (.not.active) return
  if(lo/=1.or.hi/=n) error stop 'GLACIER_LEDGER requires serial tile execution'
  do k=1,n
   if(count(mids==mids(k))/=1) error stop 'GLACIER_LEDGER requires unique tile MIDs'
  enddo
  inquire(file='MESH_glacier_storage.ini',exist=exists)
  finite=exists
  if(finite) then
   open(newunit=u,file='MESH_glacier_storage.ini',status='old',action='read')
   read(u,*,iostat=ios) rows
   if(ios/=0.or.rows/=count(ice)) error stop 'GLACIER_LEDGER invalid initial count'
   do k=1,rows
    read(u,*,iostat=ios) id,amount
    if(ios/=0) error stop 'GLACIER_LEDGER invalid initial row'
    if(.not.ieee_is_finite(amount).or.amount<0) error stop 'GLACIER_LEDGER invalid initial mass'
    j=findloc(mids,id,dim=1)
    if(j==0) error stop 'GLACIER_LEDGER unknown MID'
    if(.not.ice(j).or.seen(j)) error stop 'GLACIER_LEDGER duplicate/nonice MID'
    mass(j)=amount;seen(j)=.true.
   enddo
   close(u)
   if(any(ice.and..not.seen)) error stop 'GLACIER_LEDGER missing initial mass'
  endif
  startmass=mass
  if(len_trim(pending_restart)>0) then
   if(has_pending_stamp) then
    call ledger_read(trim(pending_restart),pending_stamp)
   else
    call ledger_read(trim(pending_restart))
   endif
  endif
  open(newunit=logunit,file='MESH_glacier_ledger.csv',status='replace',action='write')
  write(logunit,'(a)') 'step,tile,mid,start_kgm2,end_kgm2,melt_kgm2,vapor_loss_kgm2,'// &
     'snow_to_ice_kgm2,pond_to_snow_kgm2,melt_energy_Jm2,residual_kgm2'
 end subroutine
 subroutine ledger_begin()
  if(active) flux=0
 end subroutine
 subroutine ledger_flux(i,kind,amount)
  integer,intent(in)::i,kind
  real,intent(in)::amount
  if(.not.active) return
  if(.not.ieee_is_finite(amount)) error stop 'GLACIER_LEDGER nonfinite flux'
  flux(i,kind)=flux(i,kind)+real(amount,real64)
 end subroutine
 subroutine ledger_commit()
  integer::i,u
  real(real64)::proposed(size(mass)),residual
  if(.not.active) return
  proposed=mass+flux(:,3)-flux(:,1)-flux(:,2)
  if(finite.and.any(proposed<0.and.ice)) then
   open(newunit=u,file='MESH_glacier_depletion.txt',status='replace')
   write(u,*) 'Uncommitted step',step+1
   do i=1,size(mass)
    if(ice(i).and.proposed(i)<0) write(u,*) mids(i),mass(i),proposed(i)
   enddo
   close(u)
   if(logunit/=0) flush(logunit)
   error stop 'GLACIER_DEPLETION: exposed-bed transition unsupported; step NOT committed'
  endif
  step=step+1
  do i=1,size(mass)
   if(.not.ice(i)) cycle
   residual=proposed(i)-mass(i)-flux(i,3)+flux(i,1)+flux(i,2)
   write(logunit,'(i0,2(",",i0),8(",",es25.17))') step,i,mids(i),mass(i),proposed(i),flux(i,:),residual
  enddo
  mass=proposed;cumulative=cumulative+flux
  flush(logunit)
 end subroutine
 subroutine ledger_save(path,stamp)
  character(*),intent(in)::path
  integer,intent(in),optional::stamp(4)
  integer::u,i
  if(.not.active) return
  open(newunit=u,file=path,status='replace',action='write')
  write(u,*) 'SYMFLUENCE_GLACIER_V1'
  write(u,*) size(mass),step,finite
  do i=1,size(mass)
   write(u,'(i0,7(1x,es25.17))') mids(i),mass(i),startmass(i),cumulative(i,:)
  enddo
  if(present(stamp)) write(u,*) stamp
  close(u)
 end subroutine
 subroutine ledger_read(path,stamp)
  character(*),intent(in)::path
  integer,intent(in),optional::stamp(4)
  integer::u,i,n,version_step,id,ios
  integer::saved_stamp(4)
  logical::f
  character(64)::magic
  real(real64)::m,s,c(5)
  if(.not.allocated(mass)) then
   if(present(stamp)) then
    pending_stamp=stamp;has_pending_stamp=.true.
   endif
   pending_restart=path;return
  endif
  if(.not.active) return
  open(newunit=u,file=path,status='old',action='read',iostat=ios)
  if(ios/=0) error stop 'GLACIER_LEDGER missing restart sidecar'
  read(u,*,iostat=ios) magic
  if(ios/=0.or.trim(magic)/='SYMFLUENCE_GLACIER_V1') error stop 'GLACIER_LEDGER restart version'
  read(u,*,iostat=ios) n,version_step,f
  if(ios/=0.or.n/=size(mass).or.(f.neqv.finite)) error stop 'GLACIER_LEDGER restart layout'
  do i=1,n
   read(u,*,iostat=ios) id,m,s,c
   if(ios/=0.or.id/=mids(i)) error stop 'GLACIER_LEDGER restart identity'
   if(.not.all(ieee_is_finite([m,s,c]))) error stop 'GLACIER_LEDGER restart nonfinite'
   if(finite.and.m<0) error stop 'GLACIER_LEDGER negative restart'
   mass(i)=m;startmass(i)=s;cumulative(i,:)=c
  enddo
  if(present(stamp)) then
   read(u,*,iostat=ios) saved_stamp
   if(ios/=0) error stop 'GLACIER_LEDGER missing restart clock'
   if(any(saved_stamp/=stamp)) error stop 'GLACIER_LEDGER restart date must match forcing start'
  endif
  step=version_step;close(u)
 end subroutine
end module
