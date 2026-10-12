// Copyright 2026 Yellowmoon contributors. Licensed under the Apache License, Version 2.0.
//
// Stand-in for the DNG SDK's dng_jxl.cpp, used by build_dng_validate.sh.
//
// The SDK's JPEG XL support needs libjxl 0.10 or newer and the XMP toolkit. Yellowmoon DNGs never
// contain JPEG XL data (their raw image is uncompressed LinearRaw and the preview is a baseline
// JPEG), and dng_validate is only run on them, so the JPEG XL entry points are replaced with
// versions that report "not supported" and the build only needs the JPEG XL *headers*.

#include "dng_jxl.h"

#include "dng_exceptions.h"
#include "dng_host.h"
#include "dng_stream.h"

bool ParseJXL (dng_host &, dng_stream &, dng_info &, bool, bool)
	{
	return false;
	}

dng_jxl_decoder::~dng_jxl_decoder ()
	{
	}

void dng_jxl_decoder::Decode (dng_host &, dng_stream &)
	{
	ThrowNotYetImplemented ("JPEG XL (disabled in this dng_validate build)");
	}

void dng_jxl_decoder::ProcessExifBox (dng_host &, const std::vector<uint8> &)
	{
	}

void dng_jxl_decoder::ProcessXMPBox (dng_host &, const std::vector<uint8> &)
	{
	}

void dng_jxl_decoder::ProcessBox (dng_host &, const dng_string &, const std::vector<uint8> &)
	{
	}

void EncodeJXL_Tile (dng_host &, dng_stream &, const dng_pixel_buffer &,
					 const dng_jxl_color_space_info &, const dng_jxl_encode_settings &)
	{
	ThrowNotYetImplemented ("JPEG XL (disabled in this dng_validate build)");
	}

void EncodeJXL_Tile (dng_host &, dng_stream &, const dng_image &,
					 const dng_jxl_color_space_info &, const dng_jxl_encode_settings &)
	{
	ThrowNotYetImplemented ("JPEG XL (disabled in this dng_validate build)");
	}

void EncodeJXL_Container (dng_host &, dng_stream &, const dng_image &,
						  const dng_jxl_encode_settings &, const dng_jxl_color_space_info &,
						  const dng_metadata *, const bool, const bool, const bool,
						  const dng_bmff_box_list *)
	{
	ThrowNotYetImplemented ("JPEG XL (disabled in this dng_validate build)");
	}

void EncodeJXL_Container (dng_host &, dng_stream &, const dng_pixel_buffer &,
						  const dng_jxl_encode_settings &, const dng_jxl_color_space_info &,
						  const dng_metadata *, const bool, const bool, const bool,
						  const dng_bmff_box_list *)
	{
	ThrowNotYetImplemented ("JPEG XL (disabled in this dng_validate build)");
	}

real32 JXLQualityToDistance (uint32)
	{
	return 1.0f;
	}

dng_jxl_encode_settings * JXLQualityToSettings (uint32)
	{
	return new dng_jxl_encode_settings;
	}

void PreviewColorSpaceToJXLEncoding (const PreviewColorSpaceEnum, const uint32, dng_jxl_color_space_info &)
	{
	}

bool SupportsJXL (const dng_image &)
	{
	return false;
	}
