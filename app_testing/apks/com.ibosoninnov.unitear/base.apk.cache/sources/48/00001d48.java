package com.google.ar.sceneform.utilities;

import android.content.res.AssetManager;
import android.util.Log;
import c.b.a.a.a;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionException;

/* loaded from: classes.dex */
public final class SceneformBufferUtils {
    private static final int DEFAULT_BLOCK_SIZE = 8192;
    private static final String TAG = "SceneformBufferUtils";

    private SceneformBufferUtils() {
    }

    private static int copy(InputStream inputStream, OutputStream outputStream) {
        byte[] bArr = new byte[8192];
        int i = 0;
        while (true) {
            int read = inputStream.read(bArr);
            if (read > 0) {
                i += read;
                outputStream.write(bArr, 0, read);
            } else {
                outputStream.flush();
                return i;
            }
        }
    }

    public static ByteBuffer copyByteBuffer(ByteBuffer byteBuffer) {
        return ByteBuffer.wrap(copyByteBufferToArray(byteBuffer));
    }

    public static byte[] copyByteBufferToArray(ByteBuffer byteBuffer) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        byte[] bArr = new byte[8192];
        int limit = byteBuffer.limit();
        while (byteBuffer.position() < limit) {
            int position = byteBuffer.position();
            byteBuffer.get(bArr, 0, Math.min(8192, limit - position));
            byteArrayOutputStream.write(bArr, 0, byteBuffer.position() - position);
        }
        byteArrayOutputStream.flush();
        return byteArrayOutputStream.toByteArray();
    }

    public static byte[] inputStreamCallableToByteArray(Callable<InputStream> callable) {
        InputStream call = callable.call();
        try {
            byte[] inputStreamToByteArray = inputStreamToByteArray(call);
            if (call != null) {
                call.close();
            }
            return inputStreamToByteArray;
        } catch (Throwable th) {
            if (call != null) {
                try {
                    call.close();
                } catch (Throwable th2) {
                    th.addSuppressed(th2);
                }
            }
            throw th;
        }
    }

    public static byte[] inputStreamToByteArray(InputStream inputStream) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        copy(inputStream, byteArrayOutputStream);
        return byteArrayOutputStream.toByteArray();
    }

    public static ByteBuffer inputStreamToByteBuffer(Callable<InputStream> callable) {
        try {
            InputStream call = callable.call();
            ByteBuffer readStream = readStream(call);
            if (call != null) {
                call.close();
            }
            if (readStream != null) {
                return readStream;
            }
            throw new AssertionError("Failed reading data from stream");
        } catch (Exception e2) {
            throw new CompletionException(e2);
        }
    }

    public static ByteBuffer readFile(AssetManager assetManager, String str) {
        try {
            InputStream open = assetManager.open(str);
            ByteBuffer readStream = readStream(open);
            if (open != null) {
                try {
                    open.close();
                } catch (IOException e2) {
                    String str2 = TAG;
                    StringBuilder B = a.B("Failed to close the input stream from file ", str, " - ");
                    B.append(e2.getMessage());
                    Log.e(str2, B.toString());
                }
            }
            return readStream;
        } catch (IOException e3) {
            String str3 = TAG;
            StringBuilder B2 = a.B("Failed to read file ", str, " - ");
            B2.append(e3.getMessage());
            Log.e(str3, B2.toString());
            return null;
        }
    }

    public static ByteBuffer readStream(InputStream inputStream) {
        if (inputStream == null) {
            return null;
        }
        try {
            return ByteBuffer.wrap(inputStreamToByteArray(inputStream));
        } catch (IOException e2) {
            String str = TAG;
            StringBuilder x = a.x("Failed to read stream - ");
            x.append(e2.getMessage());
            Log.e(str, x.toString());
            return null;
        }
    }
}