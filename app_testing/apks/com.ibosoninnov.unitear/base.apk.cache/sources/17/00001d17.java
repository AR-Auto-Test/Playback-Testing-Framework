package com.google.ar.sceneform.rendering;

import android.content.Context;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;

/* loaded from: classes.dex */
public class ResourceHelper {
    public static ByteBuffer readResource(Context context, int i) {
        ByteBuffer byteBuffer = null;
        if (context != null) {
            int i2 = 0;
            try {
                InputStream openRawResource = context.getResources().openRawResource(i);
                openRawResource.mark(-1);
                while (openRawResource.read() != -1) {
                    i2++;
                }
                openRawResource.reset();
                ByteBuffer allocateDirect = ByteBuffer.allocateDirect(i2);
                ReadableByteChannel newChannel = Channels.newChannel(openRawResource);
                newChannel.read(allocateDirect);
                newChannel.close();
                allocateDirect.rewind();
                byteBuffer = allocateDirect;
                return byteBuffer;
            } catch (IOException e2) {
                e2.printStackTrace();
                return byteBuffer;
            }
        }
        return null;
    }
}