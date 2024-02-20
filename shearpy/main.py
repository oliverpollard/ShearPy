import cartopy
import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
import xarray as xr
from matplotlib.patches import Patch
from affine import Affine
from rasterio.features import rasterize

from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

def poly_to_grid(
    polygons,
    grid_x,
    grid_y,
    fill_value=0,
    polygon_values=None,
    nonuniform_x=False,
    nonuniform_y=False,
):

    grid_x_len = len(grid_x)
    grid_y_len = len(grid_y)

    # Affine transform
    # diagonals : a, e; shears : b, d; offsets : c, f
    if not nonuniform_y:
        dy = np.abs(grid_y[1] - grid_y[0])
    else:
        dy = np.mean(np.diff(grid_y))
    if not nonuniform_x:
        dx = np.abs(grid_x[1] - grid_x[0])
    else:
        dx = np.mean(np.diff(grid_x))

    x_off = np.min(grid_x)
    y_off = np.min(grid_y)

    x_shear = y_shear = 0

    if isinstance(polygons, gpd.geodataframe.GeoDataFrame):
        if polygons.empty:
            polygons = None
        else:
            polygons_gdf = polygons.copy()
            polygons = []
            for _, region in polygons_gdf.iterrows():
                polygons.append(region["geometry"])

    if polygons is not None:
        if not isinstance(
            polygons, (list, tuple, np.ndarray, gpd.geodataframe.GeoDataFrame)
        ):
            polygons = [polygons]

        if polygon_values is None:
            polygon_values = 1

        if isinstance(polygon_values, (list, tuple, np.ndarray)):
            assert len(polygon_values) == len(polygons)
        else:
            polygon_values = [polygon_values] * len(polygons)

        polygon_list = [
            (polygons[index], polygon_values[index]) for index in range(len(polygons))
        ]

        if nonuniform_x and nonuniform_y:
            raise NotImplementedError("Not yet impliment both nonuniform x and y.")

        elif nonuniform_y:
            polygon_gridded = np.zeros((grid_y_len, grid_x_len))
            for index in range(len(grid_y)):
                y_off = grid_y[index]
                transform = Affine(dx, x_shear, x_off, y_shear, dy, y_off)
                polygon_gridded[index] = rasterize(
                    polygons,
                    out_shape=(1, grid_x_len),
                    transform=transform,
                    fill=fill_value,
                    all_touched=True,
                )

        elif nonuniform_x:
            polygon_gridded = np.zeros((grid_y_len, grid_x_len))
            for index in range(len(grid_x)):
                x_off = grid_x[index]
                transform = Affine(dx, x_shear, x_off, y_shear, dy, y_off)
                polygon_gridded[:, index] = rasterize(
                    polygons,
                    out_shape=(grid_y_len, 1),
                    transform=transform,
                    fill=fill_value,
                    all_touched=True,
                )[:, 0]
        else:
            transform = Affine(dx, x_shear, x_off, y_shear, dy, y_off)
            polygon_gridded = rasterize(
                polygon_list,
                out_shape=(grid_y_len, grid_x_len),
                transform=transform,
                fill=fill_value,
                all_touched=True,
            )
    else:
        polygon_gridded = np.zeros((grid_y_len, grid_x_len))

    return polygon_gridded


def generate_shear_stress_mask(gdf, grid_x, grid_y):
    category_mapping = {
        category: value + 1 for value, category in enumerate(gdf["category"].unique())
    }
    shapes = []
    polygon_values = []
    for _, region in gdf.iterrows():
        shapes.append(region["geometry"])
        polygon_values.append(category_mapping[region["category"]])

    mask_raster = poly_to_grid(shapes, grid_x, grid_y, polygon_values=polygon_values)

    return mask_raster, category_mapping


def generate_region_mask(regions_gdf, grid_x, grid_y, categories=None):
    if categories is None:
        category_mapping = {
            category: value + 1
            for value, category in enumerate(regions_gdf["category"].unique())
        }
    else:
        category_mapping = {
            category: value + 1 for value, category in enumerate(categories)
        }

    shapes = []
    polygon_values = []
    for _, region in regions_gdf.iterrows():
        if region["category"] in category_mapping.keys():
            shapes.append(region["geometry"])
            polygon_values.append(category_mapping[region["category"]])

    mask_raster = poly_to_grid(shapes, grid_x, grid_y, polygon_values=polygon_values)

    return mask_raster, category_mapping


def generate_cold_ice_mask(
    ice_margin,
    grid_x,
    grid_y,
    interior_dist,
    smooth_dist=None,
    order=1,
    resolution=None,
):
    interior_dist = -interior_dist * 10**3
    shapes = []
    polygon_values = []
    if order == 0:
        buffer = ice_margin.buffer(interior_dist)
        for shape in buffer:
            if not shape.is_empty:
                shapes.append(shape)
                polygon_values.append(1)

    else:
        smooth_dist = -smooth_dist * 10**3
        distance_space = np.linspace(interior_dist, smooth_dist, num=resolution)
        value_space = np.linspace(0, 1, num=resolution) ** order
        for index in tqdm(range(len(distance_space))):
            shapes.append(ice_margin.buffer(distance_space[index])[0])
            polygon_values.append(value_space[index])

    if shapes and (not np.all([s.is_empty for s in shapes])):
        mask_raster = poly_to_grid(
            shapes, grid_x, grid_y, polygon_values=polygon_values
        )
    else:
        mask_raster = np.zeros((len(grid_y), len(grid_x)))

    return mask_raster


def generate_ice_stream_gradient_mask(
    ice_margin, grid_x, grid_y, interior_dist, resolution=30, order=1
):
    interior_dist = -interior_dist * 10**3
    shapes = []
    polygon_values = []
    if order == 0:
        outer_buffer = ice_margin.buffer(0)
        for shape in outer_buffer:
            if not shape.is_empty:
                shapes.append(shape)
                polygon_values.append(1)
        inner_buffer = ice_margin.buffer(interior_dist)
        for shape in inner_buffer:
            if not shape.is_empty:
                shapes.append(shape)
                polygon_values.append(0)
    else:
        distance_space = np.linspace(0, interior_dist, num=resolution)
        value_space = np.flip(np.linspace(0, 1, num=resolution)) ** order
        for index in tqdm(range(len(distance_space))):
            shapes.append(ice_margin.buffer(distance_space[index])[0])
            polygon_values.append(value_space[index])
    mask_raster = poly_to_grid(shapes, grid_x, grid_y, polygon_values=polygon_values)

    return mask_raster


def generate_hybrid_ice_stream(ice_margin, dist):
    landmass = list(cartopy.feature.LAND.geometries())[112]
    margin_buffer = (
        ice_margin.buffer(-dist * 10**3).symmetric_difference(ice_margin).to_crs(4326)
    )
    margin_intersect = (
        margin_buffer.intersection(landmass).to_crs(ice_margin.crs).explode()
    )
    max_area_index = np.argmax(margin_intersect.area)
    geom = margin_intersect[0][max_area_index]

    return geom


class ShearStressData:
    def __init__(self, gdf):
        self.gdf = gdf
        self.crs = gdf.crs

    def plot(
        self,
        ax=None,
        plot_crs=None,
        extent=None,
        categories=None,
        layers=None,
        ids=None,
        category_args=None,
        save=None,
        region_ids=False,
        legend_elements=None,
    ):
        if plot_crs is None:
            plot_crs = ccrs.LambertAzimuthalEqualArea(central_latitude=90)

        print(self.gdf.crs)
        gdf = self.gdf.to_crs(plot_crs.proj4_init)

        if extent is None:
            extent = [-1265453.0, 4159547.0, -4722734.8, 1352265.2]

        if ax is None:
            fig, ax = plot.plot_map(plot_crs=plot_crs, ax=None, extent=extent)

        if (layers is None) and (ids is None):
            layers = self.layers

        if categories is None:
            if layers is not None:
                categories = self.layer_categories(layers)
            else:
                categories = self.id_categories(ids)

        legend_elements = []
        for category in categories:
            label = category
            color = "green"
            if category_args is not None:
                if category in category_args:
                    if "label" in category_args[category]:
                        label = category_args[category]["label"]
                    if "color" in category_args[category]:
                        color = category_args[category]["color"]
            if layers is not None:
                for layer in layers:
                    gdf_layer = gdf[gdf["layer"] == layer]
                    gdf_layer[gdf_layer["category"] == category].plot(
                        ax=ax, color=color, transform=plot_crs, zorder=10
                    )
            else:
                for id in ids:
                    gdf_layer = gdf[gdf.index == id]
                    gdf_layer[gdf_layer["category"] == category].plot(
                        ax=ax, color=color, transform=plot_crs, zorder=10
                    )
            legend_elements.append(Patch(facecolor=color, label=label))

        if region_ids is True:
            for idx, row in enumerate(self._polygon_centers(gdf=gdf)):
                if (gdf.iloc[idx]["category"] in categories) and (
                    gdf.iloc[idx]["layer"] in layers
                ):
                    ax.text(
                        row[0],
                        row[1],
                        s=gdf.index[idx],
                        horizontalalignment="center",
                        bbox={
                            "facecolor": "white",
                            "alpha": 0.8,
                            "pad": 2,
                            "edgecolor": "none",
                        },
                        zorder=100,
                    )

        ax.legend(handles=legend_elements, loc="upper left", fontsize=12, frameon=True)

        if save is not None:
            fig.savefig(save, dpi=300, bbox_inches="tight")
        else:
            return ax

    def _polygon_centers(self, gdf=None):
        if gdf is None:
            gdf = self.gdf
        polygon_centers = gdf["geometry"].apply(
            lambda x: x.representative_point().coords[:]
        )
        polygon_centers = [coords[0] for coords in polygon_centers]
        return polygon_centers

    def to_mask(self, grid_x, grid_y, projection, categories=None, layers=None):
        category_conditional = self.gdf["category"].copy()
        if categories is not None:
            category_conditional[:] = False
            for category in categories:
                category_conditional = category_conditional | (
                    self.gdf["category"] == category
                )

        else:
            category_conditional[:] = True

        layer_conditional = self.gdf["layer"].copy()
        if layers is not None:
            layer_conditional[:] = False
            for layer in layers:
                layer_conditional = layer_conditional | (self.gdf["layer"] == layer)

        else:
            layer_conditional[:] = True

        conditional = category_conditional & layer_conditional

        gdf = self.gdf[conditional].to_crs(projection)

        mask, mask_mapping = generate_region_mask(gdf, grid_x, grid_y)
        raster_mask = RasterMask(
            mask=mask,
            mask_mapping=mask_mapping,
            grid_x=grid_x,
            grid_y=grid_y,
            projection=projection,
        )
        return raster_mask

    def to_shp(self, filename):
        self.gdf.to_file(driver="ESRI Shapefile", filename=filename)

    def recategorise(self, ids, category):
        for id in ids:
            self.gdf.loc[id - 1, "category"] = category

    def add_region(self, category, layer, geometry):
        region = self.gdf.loc[0].copy()
        region.at["category"] = category
        region.at["layer"] = layer
        region.at["geometry"] = geometry

        self.gdf = self.gdf.append(region, ignore_index=True)
        self.gdf.set_crs(self.crs, inplace=True)
        self.gdf.reset_index(drop=True)

    def layer_categories(self, layers):
        categories = []
        for layer in layers:
            for category in list(
                self.gdf[self.gdf["layer"] == layer]["category"].unique()
            ):
                if category not in categories:
                    categories.append(category)

        return categories

    def id_categories(self, ids):
        categories = []
        for id in ids:
            for category in list(self.gdf[self.gdf.index == id]["category"].unique()):
                if category not in categories:
                    categories.append(category)

        return categories

    def category_ids(self, category, layer=None):
        return np.array(self.gdf[self.gdf["category"] == category].index)

    @property
    def categories(self):
        return list(self.gdf["category"].unique())

    @property
    def layers(self):
        return list(self.gdf["layer"].unique())

    @classmethod
    def from_shp(cls, shp_file):
        gdf = gpd.read_file(shp_file)
        return cls(gdf=gdf)


class RasterMask:
    def __init__(self, mask, mask_mapping, grid_x, grid_y, projection):
        self.mask = mask
        self.mask_mapping = mask_mapping
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.projection = projection

    def plot(self):
        pass

    def to_raster(self, category_values):
        raster = np.zeros_like(self.mask, dtype=float)
        for name, value in category_values.items():
            if name in self.mask_mapping:
                key = self.mask_mapping[name]
                raster[self.mask == key] = value

        raster_type = "categorical"
        parameters = category_values
        raster_layer = RasterLayer(
            raster=raster,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            projection=self.projection,
            raster_type=raster_type,
            parameters=parameters,
        )

        return raster_layer

    @property
    def categories(self):
        return list(self.mask_mapping.keys())


class RasterLayer:
    def __init__(
        self, raster, grid_x, grid_y, projection, raster_type=None, parameters=None
    ):
        self.raster = raster
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.projection = projection
        self.raster_type = raster_type
        if parameters is None:
            parameters = {}
        self.parameters = parameters

    def __add__(self, other):
        raster = self.raster.copy() + other.raster.copy()
        raster_layer = RasterLayer(
            raster=raster,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            projection=self.projection,
            raster_type="combination",
        )
        return raster_layer

    __radd__ = __add__

    def __sub__(self, other):
        raster = self.raster.copy() - other.raster.copy()
        raster_layer = RasterLayer(
            raster=raster,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            projection=self.projection,
            raster_type="combination",
        )
        return raster_layer

    __rsub__ = __sub__

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            raster = self.raster.copy() * other
        else:
            raster = self.raster.copy() * other.raster.copy()
        raster_layer = RasterLayer(
            raster=raster,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            projection=self.projection,
            raster_type="combination",
        )
        return raster_layer

    __rmul__ = __mul__

    def __or__(self, other):
        raster = self.raster.copy()
        other_raster = other.raster.copy()
        raster[other_raster > 0] = other_raster[other_raster > 0]
        raster_layer = RasterLayer(
            raster=raster,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            projection=self.projection,
            raster_type="combination",
        )
        return raster_layer

    def plot(self, ax=None, plot_crs=None, extent=None, cmap=None, save=None):
        if plot_crs is None:
            plot_crs = ccrs.LambertAzimuthalEqualArea(central_latitude=90)
        if extent is None:
            extent = [-1265453.0, 4159547.0, 1352265.2, -4722734.8]

        if ax is None:
            fig, ax = plot.plot_map(plot_crs=plot_crs, ax=None, extent=extent)

        if cmap is None:
            cmap = resources.cmaps["shearstress"]

        min_val = np.min(np.ma.masked_where(self.raster == 0, self.raster))
        ax.imshow(
            np.flipud(np.ma.masked_where(self.raster < min_val, self.raster)),
            cmap=cmap,
            origin="upper",
            interpolation="none",
            transform=plot_crs,
            zorder=10,
            extent=[
                np.min(self.grid_x),
                np.max(self.grid_x),
                np.min(self.grid_y),
                np.max(self.grid_y),
            ],
        )
        """
        cax = fig.add_axes(
            [
                ax.get_position().x1 + 0.03,
                ax.get_position().y0,
                0.02,
                ax.get_position().height,
            ]
        )
        fig.colorbar(
            mappable,
            cax=cax,
            # ticks=np.linspace(0, 150, num=7),
            label="Shear Stress (k Pa)",
        )
        """
        return ax

    def gauss_blur(self, sigma):
        self.raster = gaussian_filter(self.raster, sigma=sigma)
        self.parameters["gauss_blur"] = sigma

    def max_value(self, value):
        self.raster[self.raster > value] = value

    def min_value(self, value):
        self.raster[self.raster < value] = value

    def base_value(self, value):
        self.raster[self.raster == 0] = value

    def to_netcdf(self, output_file):
        x_attrs = {
            "units": "meters",
            "axis": "X",
        }
        y_attrs = {
            "units": "meters",
            "axis": "Y",
        }
        # variable attributes
        shear_stress_attrs = {
            "units": "Pa",
            "long_name": "Shear stress",
            "standard_name": "Shear stress",
        }

        data_vars = {
            "z": (["y", "x"], self.raster, shear_stress_attrs),
        }
        coords = {
            "x": (["x"], self.grid_x, x_attrs),
            "y": (["y"], self.grid_y, y_attrs),
        }
        shear_stress_ds = xr.Dataset(data_vars=data_vars, coords=coords)
        shear_stress_ds.rio.write_crs(self.projection, inplace=True)
        shear_stress_ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        shear_stress_ds.to_netcdf(output_file)

    @classmethod
    def cold_ice_mask(
        cls,
        ice_margin,
        grid_x,
        grid_y,
        projection,
        interior_dist,
        smooth_dist=None,
        order=0,
        resolution=None,
    ):
        raster = generate_cold_ice_mask(
            ice_margin=ice_margin,
            grid_x=grid_x,
            grid_y=grid_y,
            interior_dist=interior_dist,
            smooth_dist=smooth_dist,
            order=order,
            resolution=resolution,
        )
        raster_type = "cold_ice_mask"
        parameters = {
            "interior_dist": interior_dist,
            "smooth_dist": smooth_dist,
            "order": order,
            "resolution": resolution,
        }
        return cls(
            raster=raster,
            grid_x=grid_x,
            grid_y=grid_y,
            projection=projection,
            raster_type=raster_type,
            parameters=parameters,
        )

    @classmethod
    def ice_stream_gradient_mask(
        cls,
        ice_margin,
        grid_x,
        grid_y,
        projection,
        interior_dist,
        order=1,
        resolution=30,
    ):
        # ice_margin = ice_margin.to_crs(projection)
        raster = generate_ice_stream_gradient_mask(
            ice_margin=ice_margin,
            grid_x=grid_x,
            grid_y=grid_y,
            interior_dist=interior_dist,
            order=order,
            resolution=resolution,
        )
        raster_type = "ice_stream_gradient_mask"
        parameters = {
            "interior_dist": interior_dist,
            "order": order,
            "resolution": resolution,
        }
        return cls(
            raster=raster,
            grid_x=grid_x,
            grid_y=grid_y,
            projection=projection,
            raster_type=raster_type,
            parameters=parameters,
        )

    @classmethod
    def hybrid_ice_stream(cls, ice_margin, grid_x, grid_y, projection, dist):
        landmass = list(cartopy.feature.LAND.geometries())[112]
        margin_buffer = (
            ice_margin.buffer(-dist * 10**3)
            .symmetric_difference(ice_margin)
            .to_crs(4326)
        ).buffer(0)
        margin_intersect = (
            margin_buffer.intersection(landmass)
            .to_crs(ice_margin.crs)
            .explode(index_parts=True)
        )
        max_area_index = np.argmax(margin_intersect.area)
        geom = margin_intersect.iloc[max_area_index]

        raster = poly_to_grid(geom, grid_x, grid_y)

        raster_type = "hybrid_ice_stream"
        parameters = {
            "dist": dist,
        }

        return cls(
            raster=raster,
            grid_x=grid_x,
            grid_y=grid_y,
            projection=projection,
            raster_type=raster_type,
            parameters=parameters,
        )


class Raster:
    def __init__(self, grid_x, grid_y, layers=None, raster=None):
        self.layers = layers
        self.raster = raster
        self.grid_x = grid_x
        self.grid_y = grid_y


def make_shear_stress_map(grid_x, grid_y, grid_crs, parameters, save=None):
    m_base_map_name = parameters["m_base_map_name"]
    m_base_map_file = parameters["m_base_map_file"]

    ss_map = ShearStressData.from_shp(m_base_map_file)

    m_gauss_blur = parameters["m_gauss_blur"]
    m_base_value = parameters["m_base_value"]
    m_combine_pattern = parameters["m_combine_pattern"]
    g_layers = parameters["g_layers"]

    raster_dict = {}

    for g_layer_name in g_layers:
        layer_categories = [
            item.replace(f"g_{g_layer_name}_", "")
            for item in list(parameters.keys())
            if item.startswith(f"g_{g_layer_name}_")
        ]
        g_layer_values = {}
        for category in layer_categories:
            g_layer_values[category] = parameters[f"g_{g_layer_name}_{category}"]

        g_layer_mask = ss_map.to_mask(grid_x, grid_y, grid_crs, layers=[g_layer_name])
        g_layer = g_layer_mask.to_raster(g_layer_values)
        raster_dict[f"g_{g_layer_name}"] = g_layer

    p_processes = parameters["p_processes"]
    if p_processes is not None:
        p_margin_name = parameters["p_margin_name"]
        p_margin_time = parameters["p_margin_time"]
        p_margin_file = parameters["p_margin_file"]

        margin = gpd.read_file(p_margin_file)

        if "ice_stream" in p_processes:
            p_ice_stream_interior_dist = parameters["p_ice_stream_interior_dist"]
            ice_stream_gradient_mask = RasterLayer.ice_stream_gradient_mask(
                ice_margin=margin.to_crs(grid_crs),
                grid_x=grid_x,
                grid_y=grid_y,
                projection=grid_crs,
                interior_dist=p_ice_stream_interior_dist,
                order=0,
                resolution=30,
            )
            raster_dict["p_ice_stream"] = ice_stream_gradient_mask
        if "cold_ice" in p_processes:
            p_cold_ice_shear_stress = parameters["p_cold_ice_shear_stress"]
            p_cold_ice_interior_dist = parameters["p_cold_ice_interior_dist"]
            p_cold_ice_smoothness = parameters.get("p_cold_ice_smoothness", 0)

            cold_ice_mask = RasterLayer.cold_ice_mask(
                ice_margin=margin.to_crs(grid_crs),
                grid_x=grid_x,
                grid_y=grid_y,
                projection=grid_crs,
                interior_dist=p_cold_ice_interior_dist,
                smooth_dist=None,
                order=0,
                resolution=None,
            )
            cold_ice_layer = cold_ice_mask * p_cold_ice_shear_stress
            raster_dict["p_cold_ice"] = cold_ice_layer

        if "hybrid_ice_stream" in p_processes:
            p_hybrid_ice_stream_shear_stress = parameters[
                "p_hybrid_ice_stream_shear_stress"
            ]
            p_hybrid_ice_stream_dist = parameters["p_hybrid_ice_stream_dist"]

            hybrid_ice_stream_mask = RasterLayer.hybrid_ice_stream(
                ice_margin=margin.to_crs(grid_crs),
                grid_x=grid_x,
                grid_y=grid_y,
                projection=grid_crs,
                dist=p_hybrid_ice_stream_dist,
            )
            hybrid_ice_stream_layer = (
                hybrid_ice_stream_mask * p_hybrid_ice_stream_shear_stress
            )
            raster_dict["p_hybrid_ice_stream"] = hybrid_ice_stream_layer

    m_combine_pattern = parameters["m_combine_pattern"]
    for g_layer in parameters["g_layers"]:
        m_combine_pattern = m_combine_pattern.replace(
            f"g_{g_layer}", f'raster_dict["g_{g_layer}"]'
        )
    if p_processes is not None:
        for p_process in parameters["p_processes"]:
            m_combine_pattern = m_combine_pattern.replace(
                f"p_{p_process}", f'raster_dict["p_{p_process}"]'
            )
    total_layer = eval(m_combine_pattern)
    total_layer.min_value(m_base_value)
    total_layer.gauss_blur(m_gauss_blur)
    if save:
        total_layer.to_netcdf(save)
    return total_layer
