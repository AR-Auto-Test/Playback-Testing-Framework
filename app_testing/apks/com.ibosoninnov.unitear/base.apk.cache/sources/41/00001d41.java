package com.google.ar.sceneform.utilities;

import android.content.Context;
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.net.Uri;
import android.net.http.HttpResponseCache;
import android.text.TextUtils;
import android.util.Base64;
import android.util.Log;
import c.b.a.a.a;
import com.google.ar.sceneform.utilities.LoadHelper;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLConnection;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;

/* loaded from: classes.dex */
public class LoadHelper {
    private static final String ANDROID_ASSET = "/android_asset/";
    private static final long DEFAULT_CACHE_SIZE_BYTES = 536870912;
    private static final String DRAWABLE_RESOURCE_TYPE = "drawable";
    public static final int INVALID_RESOURCE_IDENTIFIER = 0;
    private static final String RAW_RESOURCE_TYPE = "raw";
    private static final char SLASH_DELIMETER = '/';
    private static final String TAG = "com.google.ar.sceneform.utilities.LoadHelper";

    private LoadHelper() {
    }

    private static Callable<InputStream> androidResourceUriToInputStreamCreator(final Context context, final Uri uri) {
        String path = uri.getPath();
        String substring = path.substring(1, path.lastIndexOf(47));
        if (!substring.equals(RAW_RESOURCE_TYPE) && !substring.equals(DRAWABLE_RESOURCE_TYPE)) {
            throw new IllegalArgumentException("Unknown resource resourceType '" + substring + "' in uri '" + uri + "'. Resource will not be loaded");
        }
        return new Callable() { // from class: c.d.b.a.s.d
            @Override // java.util.concurrent.Callable
            public final Object call() {
                Context context2 = context;
                return context2.getContentResolver().openInputStream(uri);
            }
        };
    }

    private static boolean assetExists(AssetManager assetManager, String str) {
        String[] list;
        int lastIndexOf = str.lastIndexOf(47);
        if (lastIndexOf != -1) {
            String substring = str.substring(lastIndexOf + 1);
            list = assetManager.list(str.substring(0, lastIndexOf));
            str = substring;
        } else {
            list = assetManager.list("");
        }
        if (list != null) {
            for (String str2 : list) {
                if (str.equals(str2)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static Callable<InputStream> dataUriInputStreamCreator(Uri uri) {
        String schemeSpecificPart = uri.getSchemeSpecificPart();
        int indexOf = schemeSpecificPart.indexOf(44);
        if (indexOf >= 0) {
            final boolean contains = schemeSpecificPart.substring(0, indexOf).contains(";base64");
            final String substring = schemeSpecificPart.substring(indexOf + 1);
            return new Callable() { // from class: c.d.b.a.s.c
                @Override // java.util.concurrent.Callable
                public final Object call() {
                    boolean z = contains;
                    String str = substring;
                    return new ByteArrayInputStream(z ? Base64.decode(str, 0) : str.getBytes());
                }
            };
        }
        throw new IllegalArgumentException("Malformed data uri - does not contain a ','");
    }

    public static int drawableResourceNameToIdentifier(Context context, String str) {
        return context.getResources().getIdentifier(str, DRAWABLE_RESOURCE_TYPE, context.getPackageName());
    }

    public static void enableCaching(Context context) {
        enableCaching(DEFAULT_CACHE_SIZE_BYTES, context.getCacheDir(), "http_cache");
    }

    private static Callable<InputStream> fileUriToInputStreamCreator(Context context, Uri uri) {
        final String str;
        final AssetManager assets = context.getAssets();
        if (uri.getAuthority() == null) {
            str = uri.getPath();
        } else if (uri.getPath().isEmpty()) {
            str = uri.getAuthority();
        } else {
            str = uri.getAuthority() + uri.getPath();
        }
        final String removeAndroidAssetPath = removeAndroidAssetPath(str);
        return new Callable() { // from class: c.d.b.a.s.b
            @Override // java.util.concurrent.Callable
            public final Object call() {
                return LoadHelper.lambda$fileUriToInputStreamCreator$1(assets, removeAndroidAssetPath, str);
            }
        };
    }

    public static void flushHttpCache() {
        HttpResponseCache installed = HttpResponseCache.getInstalled();
        if (installed != null) {
            installed.flush();
        }
    }

    public static Callable<InputStream> fromResource(final Context context, final int i) {
        Preconditions.checkNotNull(context, "Parameter \"context\" was null.");
        String resourceTypeName = context.getResources().getResourceTypeName(i);
        if (!resourceTypeName.equals(RAW_RESOURCE_TYPE) && !resourceTypeName.equals(DRAWABLE_RESOURCE_TYPE)) {
            StringBuilder B = a.B("Unknown resource resourceType '", resourceTypeName, "' in resId '");
            B.append(context.getResources().getResourceName(i));
            B.append("'. Resource will not be loaded");
            throw new IllegalArgumentException(B.toString());
        }
        return new Callable() { // from class: c.d.b.a.s.e
            @Override // java.util.concurrent.Callable
            public final Object call() {
                Context context2 = context;
                return context2.getResources().openRawResource(i);
            }
        };
    }

    public static Callable<InputStream> fromUri(Context context, Uri uri) {
        return fromUri(context, uri, null);
    }

    private static String getGltfExtensionFromSchemeSpecificPart(String str) {
        if (str.startsWith("model/gltf-binary")) {
            return "glb";
        }
        if (str.startsWith("model/gltf+json")) {
            return "gltf";
        }
        return null;
    }

    public static String getLastPathSegment(Uri uri) {
        if (isGltfDataUri(uri)) {
            StringBuilder x = a.x("file.");
            x.append(getGltfExtensionFromSchemeSpecificPart(uri.getSchemeSpecificPart()));
            return x.toString();
        }
        String lastPathSegment = uri.getLastPathSegment();
        if (lastPathSegment == null) {
            String uri2 = uri.toString();
            return uri2.substring(uri2.lastIndexOf(47) + 1);
        }
        return lastPathSegment;
    }

    public static Boolean isAndroidResource(Uri uri) {
        Preconditions.checkNotNull(uri, "Parameter \"sourceUri\" was null.");
        return Boolean.valueOf(TextUtils.equals("android.resource", uri.getScheme()));
    }

    public static boolean isDataUri(Uri uri) {
        String scheme = uri.getScheme();
        return scheme != null && scheme.equals("data");
    }

    public static Boolean isFileAsset(Uri uri) {
        Preconditions.checkNotNull(uri, "Parameter \"sourceUri\" was null.");
        String scheme = uri.getScheme();
        return Boolean.valueOf(TextUtils.isEmpty(scheme) || Objects.equals("file", scheme));
    }

    public static boolean isGltfDataUri(Uri uri) {
        return isDataUri(uri) && getGltfExtensionFromSchemeSpecificPart(uri.getSchemeSpecificPart()) != null;
    }

    public static /* synthetic */ InputStream lambda$fileUriToInputStreamCreator$1(AssetManager assetManager, String str, String str2) {
        if (assetExists(assetManager, str)) {
            return assetManager.open(str);
        }
        return new FileInputStream(new File(str2));
    }

    public static int rawResourceNameToIdentifier(Context context, String str) {
        return context.getResources().getIdentifier(str, RAW_RESOURCE_TYPE, context.getPackageName());
    }

    private static Callable<InputStream> remoteUriToInputStreamCreator(Uri uri, Map<String, String> map) {
        try {
            final URLConnection openConnection = new URL(uri.toString()).openConnection();
            if (map != null) {
                for (Map.Entry<String, String> entry : map.entrySet()) {
                    openConnection.addRequestProperty(entry.getKey(), entry.getValue());
                }
            }
            return new Callable() { // from class: c.d.b.a.s.a
                @Override // java.util.concurrent.Callable
                public final Object call() {
                    return openConnection.getInputStream();
                }
            };
        } catch (MalformedURLException e2) {
            throw new IllegalArgumentException("Unable to parse url: '" + uri + "'", e2);
        } catch (IOException e3) {
            throw new AssertionError("Error opening url connection: '" + uri + "'", e3);
        }
    }

    private static String removeAndroidAssetPath(String str) {
        return str.startsWith(ANDROID_ASSET) ? str.substring(15) : str;
    }

    private static Uri resolve(Uri uri, Uri uri2) {
        try {
            return Uri.parse(new URI(uri.toString()).resolve(new URI(uri2.toString())).toString());
        } catch (URISyntaxException e2) {
            throw new IllegalArgumentException("Unable to parse Uri.", e2);
        }
    }

    public static Uri resolveUri(Uri uri, Uri uri2) {
        return uri2 == null ? uri : resolve(uri2, uri);
    }

    public static Uri resourceToUri(Context context, int i) {
        Resources resources = context.getResources();
        return new Uri.Builder().scheme("android.resource").authority(resources.getResourcePackageName(i)).appendPath(resources.getResourceTypeName(i)).appendPath(resources.getResourceEntryName(i)).build();
    }

    public static void enableCaching(long j, File file, String str) {
        if (HttpResponseCache.getInstalled() == null) {
            try {
                HttpResponseCache.install(new File(file, str), j);
            } catch (IOException e2) {
                String str2 = TAG;
                Log.i(str2, "HTTP response cache installation failed:" + e2);
            }
        }
    }

    public static Callable<InputStream> fromUri(Context context, Uri uri, Map<String, String> map) {
        Preconditions.checkNotNull(uri, "Parameter \"sourceUri\" was null.");
        Preconditions.checkNotNull(context, "Parameter \"context\" was null.");
        if (isFileAsset(uri).booleanValue()) {
            return fileUriToInputStreamCreator(context, uri);
        }
        if (isAndroidResource(uri).booleanValue()) {
            return androidResourceUriToInputStreamCreator(context, uri);
        }
        if (isGltfDataUri(uri)) {
            return dataUriInputStreamCreator(uri);
        }
        return remoteUriToInputStreamCreator(uri, map);
    }
}