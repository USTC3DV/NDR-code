//syszux.cpp
#include <pybind11/pybind11.h>
#include<pybind11/numpy.h>
namespace py = pybind11;
#include <stdio.h>
#include <iostream>

template <typename Dtype>
bool is_in_tri(Dtype *vpos, Dtype *v1pos, Dtype *v2pos, Dtype *v3pos, Dtype *tricoord)
{
    Dtype x1 = v2pos[0] - v1pos[0];
    Dtype y1 = v2pos[1] - v1pos[1];

    Dtype x2 = v3pos[0] - v1pos[0];
    Dtype y2 = v3pos[1] - v1pos[1];

    Dtype x = vpos[0] - v1pos[0];
    Dtype y = vpos[1] - v1pos[1];

    if (x1 * y2 == x2 * y1) 
    {
      return false;
    }

    Dtype b = (x * y1 - x1 * y) / (x2 * y1 - x1 * y2);
    Dtype a = (x * y2 - x2 * y) / (x1 * y2 - x2 * y1);

    tricoord[0] = Dtype(1) - a - b;
    tricoord[1] = a;
    tricoord[2] = b;

    if (a >= -0.01 && b >= -0.01 && (a + b) <= 1.01)
    {
      return true;
    }

    return false;
}

template <typename Dtype>
bool normalized_vec(Dtype *vec)
{
    Dtype tv = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]) + 1e-6;
    vec[0] /= tv;
    vec[1] /= tv;
    vec[2] /= tv;
}

template <typename Dtype>
Dtype cross_vec(Dtype *vec1, Dtype *vec2)
{
    Dtype tv = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
    return tv;
}

void render_color_mesh(
    py::array_t<float>& tri_normals, 
    py::array_t<float>& tri_colors,
    py::array_t<float>& tri_pixels,
    py::array_t<float>& tri_z_vals,
    py::array_t<bool>& tri_status,
    py::array_t<float>& depth_img,
    py::array_t<float>& rgb_img,
    py::array_t<int>& mask_img,
    float light_dx, float light_dy,float light_dz,
    float ambient_strength, float light_strength
)
{
    py::buffer_info tri_normals_buf = tri_normals.request();
    py::buffer_info tri_colors_buf = tri_colors.request();
    py::buffer_info tri_pixels_buf = tri_pixels.request();
    py::buffer_info tri_z_vals_buf = tri_z_vals.request();
    py::buffer_info tri_status_buf = tri_status.request();
    py::buffer_info depth_img_buf = depth_img.request();
    py::buffer_info rgb_img_buf = rgb_img.request();
    py::buffer_info mask_img_buf = mask_img.request();

    float* tri_normals_ptr = (float*)tri_normals_buf.ptr;
    float* tri_colors_ptr = (float*)tri_colors_buf.ptr;
    float* tri_pixels_ptr = (float*)tri_pixels_buf.ptr;
    float* tri_z_vals_ptr = (float*)tri_z_vals_buf.ptr;
    bool* tri_status_ptr = (bool*)tri_status_buf.ptr;
    float* depth_img_ptr = (float*)depth_img_buf.ptr;
    float* rgb_img_ptr = (float*)rgb_img_buf.ptr;
    int* mask_img_ptr = (int*)mask_img_buf.ptr;
    float light_dire_ptr[3] = {light_dx, light_dy, light_dz};

    int n_face = tri_normals_buf.shape[0];
    int width = depth_img_buf.shape[1];
    int height = depth_img_buf.shape[0];

    // float ambient_strength = 0.4;
    // flaot light_strength = 0.6;

    for(int fid = 0; fid < n_face; fid++)
    {
        bool f_status = tri_status_ptr[fid];
        if (!f_status)
        {
            continue;
        }

        float* Vnormals = tri_normals_ptr + fid * 9;
        float* Vcolors = tri_colors_ptr + fid * 9;
        float* Vpixels = tri_pixels_ptr + fid * 6;
        float* Vzvals = tri_z_vals_ptr + fid * 3;

        float xmin = width, xmax = -2, ymin = height, ymax = -2;
        xmin = fminf(xmin, Vpixels[0]);
        ymin = fminf(ymin, Vpixels[1]);
        xmin = fminf(xmin, Vpixels[2]);
        ymin = fminf(ymin, Vpixels[3]);
        xmin = fminf(xmin, Vpixels[4]);
        ymin = fminf(ymin, Vpixels[5]);

        xmax = fmaxf(xmax, Vpixels[0]);
        ymax = fmaxf(ymax, Vpixels[1]);
        xmax = fmaxf(xmax, Vpixels[2]);
        ymax = fmaxf(ymax, Vpixels[3]);
        xmax = fmaxf(xmax, Vpixels[4]);
        ymax = fmaxf(ymax, Vpixels[5]);

        xmin = fmaxf(float(0), xmin);
        ymin = fmaxf(float(0), ymin);
        xmax = fminf(xmax, float(width - 2));
        ymax = fminf(ymax, float(height - 2));

        float coord[3], vpos[2], temp_normal[3], temp_color[3];

        for (int x = ceilf(xmin); x <= floorf(xmax); x++) 
        {
            for (int y = ceilf(ymin); y <= floorf(ymax); y++) 
            {
                vpos[0] = x;
                vpos[1] = y;
                if (is_in_tri(vpos, &(Vpixels[0]), &(Vpixels[2]), &(Vpixels[4]), coord)) 
                {
                    int pixel_index = y * width + x;
                    float z_value = coord[0] * Vzvals[0] + coord[1] * Vzvals[1] + coord[2] * Vzvals[2];
                    float pre_z_value = depth_img_ptr[pixel_index];
                    if (z_value < pre_z_value)
                    {
                        depth_img_ptr[pixel_index] = z_value;
                        temp_normal[0] = coord[0] * Vnormals[0] + coord[1] * Vnormals[3] + coord[2] * Vnormals[6];
                        temp_normal[1] = coord[0] * Vnormals[1] + coord[1] * Vnormals[4] + coord[2] * Vnormals[7];
                        temp_normal[2] = coord[0] * Vnormals[2] + coord[1] * Vnormals[5] + coord[2] * Vnormals[8];
                        normalized_vec(temp_normal);
                        float cos_v = cross_vec(temp_normal, light_dire_ptr);
                        cos_v = fmaxf(float(0), cos_v);
                        float scale_ = (ambient_strength + light_strength * cos_v);
                        temp_color[0] = coord[0] * Vcolors[0] + coord[1] * Vcolors[3] + coord[2] * Vcolors[6];
                        temp_color[1] = coord[0] * Vcolors[1] + coord[1] * Vcolors[4] + coord[2] * Vcolors[7];
                        temp_color[2] = coord[0] * Vcolors[2] + coord[1] * Vcolors[5] + coord[2] * Vcolors[8];

                        rgb_img_ptr[3 * pixel_index] = scale_ * temp_color[0];
                        rgb_img_ptr[3 * pixel_index + 1] = scale_ * temp_color[1];
                        rgb_img_ptr[3 * pixel_index + 2] = scale_ * temp_color[2];
                    } 
                    mask_img_ptr[pixel_index] = 255;
                }
            }
        }
    }
}

void render_tex_mesh(
    py::array_t<float>& tri_normals, 
    py::array_t<float>& tri_uvs,
    py::array_t<float>& tri_pixels,
    py::array_t<float>& tri_z_vals,
    py::array_t<bool>& tri_status,
    py::array_t<float>& tex_img,
    py::array_t<float>& depth_img,
    py::array_t<float>& rgb_img,
    py::array_t<int>& mask_img,
    float light_dx, float light_dy,float light_dz,
    float ambient_strength, float light_strength
)
{
    py::buffer_info tri_normals_buf = tri_normals.request();
    py::buffer_info tri_uvs_buf = tri_uvs.request();
    py::buffer_info tri_pixels_buf = tri_pixels.request();
    py::buffer_info tri_z_vals_buf = tri_z_vals.request();
    py::buffer_info tri_status_buf = tri_status.request();
    py::buffer_info tex_img_buf = tex_img.request();
    py::buffer_info depth_img_buf = depth_img.request();
    py::buffer_info rgb_img_buf = rgb_img.request();
    py::buffer_info mask_img_buf = mask_img.request();

    float* tri_normals_ptr = (float*)tri_normals_buf.ptr;
    float* tri_uvs_ptr = (float*)tri_uvs_buf.ptr;
    float* tri_pixels_ptr = (float*)tri_pixels_buf.ptr;
    float* tri_z_vals_ptr = (float*)tri_z_vals_buf.ptr;
    bool* tri_status_ptr = (bool*)tri_status_buf.ptr;
    float* tex_img_ptr = (float*)tex_img_buf.ptr;
    float* depth_img_ptr = (float*)depth_img_buf.ptr;
    float* rgb_img_ptr = (float*)rgb_img_buf.ptr;
    int* mask_img_ptr = (int*)mask_img_buf.ptr;
    float light_dire_ptr[3] = {light_dx, light_dy, light_dz};

    int n_face = tri_normals_buf.shape[0];
    int height = depth_img_buf.shape[0];
    int width = depth_img_buf.shape[1];

    int tex_height = tex_img_buf.shape[0];
    int tex_width = tex_img_buf.shape[1];

    for(int fid = 0; fid < n_face; fid++)
    {
        bool f_status = tri_status_ptr[fid];
        if (!f_status)
        {
            continue;
        }

        float* Vnormals = tri_normals_ptr + fid * 9;
        float* Vpixels = tri_pixels_ptr + fid * 6;
        float* Vzvals = tri_z_vals_ptr + fid * 3;
        float* Vuvs = tri_uvs_ptr + fid * 6;

        float tex_triabgle_area_like = ((Vuvs[4] - Vuvs[0]) * (Vuvs[3] - Vuvs[1]) - (Vuvs[5] - Vuvs[1]) * (Vuvs[2] - Vuvs[0])) * tex_width * tex_height;
        float rgb_triabgle_area_like = (tri_pixels_ptr[4] - tri_pixels_ptr[0]) * (tri_pixels_ptr[3] - tri_pixels_ptr[1]) - (tri_pixels_ptr[5] - tri_pixels_ptr[1]) * (tri_pixels_ptr[2] - tri_pixels_ptr[0]);
        int filter_half_length_ = floor(sqrt(abs(tex_triabgle_area_like / rgb_triabgle_area_like)) * 0.5);

        float xmin = width, xmax = -2, ymin = height, ymax = -2;
        xmin = fminf(xmin, Vpixels[0]);
        ymin = fminf(ymin, Vpixels[1]);
        xmin = fminf(xmin, Vpixels[2]);
        ymin = fminf(ymin, Vpixels[3]);
        xmin = fminf(xmin, Vpixels[4]);
        ymin = fminf(ymin, Vpixels[5]);

        xmax = fmaxf(xmax, Vpixels[0]);
        ymax = fmaxf(ymax, Vpixels[1]);
        xmax = fmaxf(xmax, Vpixels[2]);
        ymax = fmaxf(ymax, Vpixels[3]);
        xmax = fmaxf(xmax, Vpixels[4]);
        ymax = fmaxf(ymax, Vpixels[5]);

        xmin = fmaxf(float(0), xmin);
        ymin = fmaxf(float(0), ymin);
        xmax = fminf(xmax, float(width - 2));
        ymax = fminf(ymax, float(height - 2));

        float coord[3], vpos[2], temp_normal[3], temp_uvs[2];

        for (int x = ceilf(xmin); x <= floorf(xmax); x++) 
        {
            for (int y = ceilf(ymin); y <= floorf(ymax); y++) 
            {
                vpos[0] = x;
                vpos[1] = y;
                if (is_in_tri(vpos, &(Vpixels[0]), &(Vpixels[2]), &(Vpixels[4]), coord)) 
                {
                    int pixel_index = y * width + x;
                    float z_value = coord[0] * Vzvals[0] + coord[1] * Vzvals[1] + coord[2] * Vzvals[2];
                    float pre_z_value = depth_img_ptr[pixel_index];
                    if (z_value < pre_z_value)
                    {
                        depth_img_ptr[pixel_index] = z_value;

                        temp_normal[0] = coord[0] * Vnormals[0] + coord[1] * Vnormals[3] + coord[2] * Vnormals[6];
                        temp_normal[1] = coord[0] * Vnormals[1] + coord[1] * Vnormals[4] + coord[2] * Vnormals[7];
                        temp_normal[2] = coord[0] * Vnormals[2] + coord[1] * Vnormals[5] + coord[2] * Vnormals[8];
                        normalized_vec(temp_normal);
                        float cos_v = cross_vec(temp_normal, light_dire_ptr);
                        cos_v = fmaxf(float(0), cos_v);
                        float scale_ = (ambient_strength + light_strength * cos_v);

                        temp_uvs[0] = coord[0] * Vuvs[0] + coord[1] * Vuvs[2] + coord[2] * Vuvs[4];
                        temp_uvs[1] = coord[0] * Vuvs[1] + coord[1] * Vuvs[3] + coord[2] * Vuvs[5];
                        
                        int tex_r = tex_height * (1.0 - temp_uvs[1]);
                        int tex_c = tex_width * temp_uvs[0];
                        
                        if (filter_half_length_ == 0)
                        {
                            float* tex_color_ptr = tex_img_ptr + 3 * (tex_r * tex_height + tex_c);
                            rgb_img_ptr[3 * pixel_index] = scale_ * tex_color_ptr[0];
                            rgb_img_ptr[3 * pixel_index + 1] = scale_ * tex_color_ptr[1];
                            rgb_img_ptr[3 * pixel_index + 2] = scale_ * tex_color_ptr[2];
                        }
                        else
                        {
                            float r_ = 0.0f;
                            float g_ = 0.0f;
                            float b_ = 0.0f;
                            float my_cnt_ = 0.0f;

                            int start_x = std::max(0, tex_c - filter_half_length_);
                            int end_x = std::min(tex_width, tex_c + filter_half_length_ + 1);

                            int start_y = std::max(0, tex_r - filter_half_length_);
                            int end_y = std::min(tex_height, tex_r + filter_half_length_ + 1);

                            for (int tx = start_x; tx < end_x; tx++)
                            {
                                for (int ty = start_y; ty < end_y; ty++)
                                {
                                    float* temp_color_ptr = tex_img_ptr + 3 * (ty * tex_height + tx);
                                    r_ += temp_color_ptr[0];
                                    g_ += temp_color_ptr[1];
                                    b_ += temp_color_ptr[2];
                                    my_cnt_++;
                                }
                            }

                            r_ *= (scale_ / my_cnt_);
                            g_ *= (scale_ / my_cnt_);
                            b_ *= (scale_ / my_cnt_);

                            rgb_img_ptr[3 * pixel_index] = r_;
                            rgb_img_ptr[3 * pixel_index + 1] = g_;
                            rgb_img_ptr[3 * pixel_index + 2] = b_;
                        }
                        mask_img_ptr[pixel_index] = 255;
                    } 
                }
            }
        }
    }
}

PYBIND11_MODULE(RenderUtils, m) 
{
    m.def("render_color_mesh", &render_color_mesh, "A function which renders colored meshes.");
    m.def("render_tex_mesh", &render_tex_mesh, "A function which renders textured meshes.");
}