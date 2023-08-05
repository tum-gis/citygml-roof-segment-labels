-------------------------------------------------------------------------------
-- Query roof segments and their attributes from 3DCityDB
-- Returns 6 columns:
--  b_id        ID of the corresponding building
--  sg_id       ID of the roof segment surface geometry (objectclass_id = 33)
--  azimuth     Azimuth of the roof segment relative to North clockwise
--  slope       Slope of the roof segment relative to the horizontal plane
--  geometry    WKT geometry of the roof segment
--  method      Number indicating the roof generation method (specific to data source)
-------------------------------------------------------------------------------

select
	b.id as "b_id",
	sg.id as "sg_id",
	-180 + degrees(
        st_azimuth(
            st_makepoint(0,0),
            st_rotate(
                st_force2d(citydb.normalvector_norm(sg.geometry)),
                pi()
            )
        )
    ) as "azimuth",
	90 + degrees(
        citydb.slope_from_normv(citydb.normalvector_norm(sg.geometry))
    ) as "slope",
	st_force2d(sg.geometry) as "geometry",
	cg.strval as "method"
from building b, thematic_surface ts, surface_geometry sg, cityobject_genericattrib cg 
where b.id = ts.building_id and
	ts.objectclass_id = 33 and
	sg.root_id = ts.lod2_multi_surface_id and
	sg.geometry is not null and
	b.id = cg.cityobject_id and
	cg.attrname = 'Methode';


-- function normal vector -----------------------------------------------------
--    Compute the normalized normal vector to the given geometry.
--    The geometry is first simplified using Visvalingam-Whyatt algorithm.
--    Tolerance is set to 9.000.000.000m.
-------------------------------------------------------------------------------
--  IN
--    ge     geometry    the geometry
--  RETURN
--    point  geometry    the normalized normal vector of the geometry represented
--    as point
-------------------------------------------------------------------------------
DROP FUNCTION IF EXISTS citydb.normalvector_norm(geometry);
CREATE OR REPLACE FUNCTION citydb.normalvector_norm(ge geometry) RETURNS geometry AS
$$
DECLARE

ux double precision;
uy double precision;
uz double precision;

vx double precision;
vy double precision;
vz double precision;

nx double precision;
ny double precision;
nz double precision;

tolerance double precision;
len double precision;

npoints integer;
points geometry[];

BEGIN
-- init vars
tolerance := 9000000000;    -- 9.000.000.000 m

-- simplify geom to triangle and dump points of exterior ring of to point array
points := (
  SELECT
    array_agg((gd).geom) geomarray
  FROM
    (SELECT ST_DumpPoints(ST_ExteriorRing(ST_SimplifyVW(ge, tolerance)))gd) gd
);

npoints := array_length(points,1);

-- compute vectors: u = 1 -> 3, v = 1 -> 2
ux = (SELECT ST_X(points[npoints-1])) - (SELECT ST_X(points[1]));
uy = (SELECT ST_Y(points[npoints-1])) - (SELECT ST_Y(points[1]));
uz = (SELECT ST_Z(points[npoints-1])) - (SELECT ST_Z(points[1]));

vx = (SELECT ST_X(points[2])) - (SELECT ST_X(points[1]));
vy = (SELECT ST_Y(points[2])) - (SELECT ST_Y(points[1]));
vz = (SELECT ST_Z(points[2])) - (SELECT ST_Z(points[1]));

-- compute normal vector with cross product: u x v = n
nx := uy*vz - uz*vy;
ny := uz*vx - ux*vz;
nz := ux*vy - uy*vx;

-- Normalize vector
len := sqrt(nx^2 + ny^2 + nz^2);
RETURN ST_MakePoint(nx/len,ny/len,nz/len);

END;
$$
LANGUAGE plpgsql;


-- function slope_from_normv(geometry) ----------------------------------------
-- computes the angle of a vector relative to the horizontal plane
-------------------------------------------------------------------------------
-- Argument:
-- ge       geometry
-- Returns:
-- slope    double precision
-------------------------------------------------------------------------------
drop function if exists citydb.slope_from_normv(geometry);
create or replace function citydb.slope_from_normv(ge geometry) returns double precision as
$$
declare 

px double precision;
py double precision;
pz double precision;
denom double precision;
angle double precision;

begin 
	px = st_x(ge);
	py = st_y(ge);
	pz = st_z(ge);

	denom = sqrt(px^2 + py^2);

	-- if at least x or y are non-zero
	if denom <> 0 then
		-- compute the slope
		angle = atan(pz/denom);
	
	-- else, i.e. if both x and y are zero, i.e. the vector points vertically up or down
	else
		-- set angle to pi/2 = 90,
		-- and set the sign according to z being positive or negative
		angle = pi()/2 * pz / abs(pz);
	end if;

	return angle;
end;
$$
language plpgsql;
