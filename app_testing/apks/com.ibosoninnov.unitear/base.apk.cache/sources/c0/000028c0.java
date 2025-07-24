package com.google.mediapipe.components;

import android.graphics.SurfaceTexture;
import android.opengl.GLES20;
import android.util.Log;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.mediapipe.framework.AppTextureFrame;
import com.google.mediapipe.framework.GlSyncToken;
import com.google.mediapipe.glutil.ExternalTextureRenderer;
import com.google.mediapipe.glutil.GlThread;
import com.google.mediapipe.glutil.ShaderUtil;
import java.lang.Thread;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import javax.microedition.khronos.egl.EGLContext;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/components/ExternalTextureConverter.class */
public class ExternalTextureConverter implements TextureFrameProducer {
    private static final String TAG = "ExternalTextureConv";
    private static final int DEFAULT_NUM_BUFFERS = 2;
    private static final String THREAD_NAME = "ExternalTextureConverter";
    private RenderThread thread;
    private Throwable startupException;

    public ExternalTextureConverter(EGLContext parentContext, int numBuffers) {
        this.startupException = null;
        this.thread = makeRenderThread(parentContext, numBuffers);
        this.thread.setName("ExternalTextureConverter");
        Object threadExceptionLock = new Object();
        this.thread.setUncaughtExceptionHandler(t, e2 -> {
            synchronized (threadExceptionLock) {
                this.startupException = e2;
                threadExceptionLock.notify();
            }
        });
        this.thread.start();
        try {
            boolean success = this.thread.waitUntilReady();
            if (!success) {
                synchronized (threadExceptionLock) {
                    while (this.startupException == null) {
                        threadExceptionLock.wait();
                    }
                }
            }
            this.thread.setUncaughtExceptionHandler(null);
            if (this.startupException != null) {
                this.thread.quitSafely();
                throw new RuntimeException(this.startupException);
            }
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            Log.e("ExternalTextureConv", "thread was unexpectedly interrupted: " + ie.getMessage());
            throw new RuntimeException(ie);
        }
    }

    public void setFlipY(boolean flip) {
        this.thread.setFlipY(flip);
    }

    public void setRotation(int rotation) {
        this.thread.setRotation(rotation);
    }

    public void setTimestampOffsetNanos(long offsetInNanos) {
        this.thread.setTimestampOffsetNanos(offsetInNanos);
    }

    public ExternalTextureConverter(EGLContext parentContext) {
        this(parentContext, 2);
    }

    public ExternalTextureConverter(EGLContext parentContext, SurfaceTexture texture, int targetWidth, int targetHeight) {
        this(parentContext);
        this.thread.setSurfaceTexture(texture, targetWidth, targetHeight);
    }

    public void setUncaughtExceptionHandler(Thread.UncaughtExceptionHandler handler) {
        this.thread.setUncaughtExceptionHandler(handler);
    }

    public void setSurfaceTexture(SurfaceTexture texture, int width, int height) {
        if (texture != null && (width == 0 || height == 0)) {
            throw new RuntimeException("ExternalTextureConverter: setSurfaceTexture dimensions cannot be zero");
        }
        this.thread.getHandler().post(()
        /*  JADX ERROR: Method code generation error
            jadx.core.utils.exceptions.CodegenException: Error generate insn: 0x0026: INVOKE  
              (wrap: android.os.Handler : 0x001a: INVOKE  (r0v3 android.os.Handler A[REMOVE]) = 
              (wrap: com.google.mediapipe.components.ExternalTextureConverter$RenderThread : 0x0017: IGET  (r0v2 com.google.mediapipe.components.ExternalTextureConverter$RenderThread A[REMOVE]) = 
              (r6v0 'this' com.google.mediapipe.components.ExternalTextureConverter A[D('this' com.google.mediapipe.components.ExternalTextureConverter), IMMUTABLE_TYPE, THIS])
             com.google.mediapipe.components.ExternalTextureConverter.thread com.google.mediapipe.components.ExternalTextureConverter$RenderThread)
             type: VIRTUAL call: com.google.mediapipe.components.ExternalTextureConverter.RenderThread.getHandler():android.os.Handler)
              (wrap: java.lang.Runnable : 0x0021: INVOKE_CUSTOM (r1v1 java.lang.Runnable A[REMOVE]) = 
              (r6v0 'this' com.google.mediapipe.components.ExternalTextureConverter A[D('this' com.google.mediapipe.components.ExternalTextureConverter), DONT_INLINE, IMMUTABLE_TYPE, THIS])
              (r7v0 'texture' android.graphics.SurfaceTexture A[D('texture' android.graphics.SurfaceTexture), DONT_INLINE])
              (r8v0 'width' int A[D('width' int), DONT_INLINE])
              (r9v0 'height' int A[D('height' int), DONT_INLINE])
            
             handle type: INVOKE_DIRECT
             lambda: java.lang.Runnable.run():void
             call insn: ?: INVOKE  
              (r1 I:com.google.mediapipe.components.ExternalTextureConverter)
              (r2 I:android.graphics.SurfaceTexture)
              (r3 I:int)
              (r4 I:int)
             type: DIRECT call: com.google.mediapipe.components.ExternalTextureConverter.lambda$setSurfaceTexture$1(android.graphics.SurfaceTexture, int, int):void)
             type: VIRTUAL call: android.os.Handler.post(java.lang.Runnable):boolean in method: com.google.mediapipe.components.ExternalTextureConverter.setSurfaceTexture(android.graphics.SurfaceTexture, int, int):void, file: base.apk:classes.jar:com/google/mediapipe/components/ExternalTextureConverter.class
            	at jadx.core.codegen.InsnGen.makeInsn(InsnGen.java:289)
            	at jadx.core.codegen.InsnGen.makeInsn(InsnGen.java:252)
            	at jadx.core.codegen.RegionGen.makeSimpleBlock(RegionGen.java:91)
            	at jadx.core.dex.nodes.IBlock.generate(IBlock.java:15)
            	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
            	at jadx.core.dex.regions.Region.generate(Region.java:35)
            	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
            	at jadx.core.dex.regions.Region.generate(Region.java:35)
            	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
            	at jadx.core.dex.regions.Region.generate(Region.java:35)
            	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
            	at jadx.core.codegen.MethodGen.addRegionInsns(MethodGen.java:296)
            	at jadx.core.codegen.MethodGen.addInstructions(MethodGen.java:275)
            	at jadx.core.codegen.ClassGen.addMethodCode(ClassGen.java:377)
            	at jadx.core.codegen.ClassGen.addMethod(ClassGen.java:306)
            	at jadx.core.codegen.ClassGen.lambda$addInnerClsAndMethods$2(ClassGen.java:272)
            	at java.util.stream.ForEachOps$ForEachOp$OfRef.accept(ForEachOps.java:184)
            	at java.util.ArrayList.forEach(ArrayList.java:1259)
            	at java.util.stream.SortedOps$RefSortingSink.end(SortedOps.java:390)
            	at java.util.stream.Sink$ChainedReference.end(Sink.java:258)
            Caused by: java.lang.IndexOutOfBoundsException: Index: 3, Size: 3
            	at java.util.ArrayList.rangeCheck(ArrayList.java:659)
            	at java.util.ArrayList.get(ArrayList.java:435)
            	at jadx.core.codegen.InsnGen.makeInlinedLambdaMethod(InsnGen.java:998)
            	at jadx.core.codegen.InsnGen.makeInvokeLambda(InsnGen.java:903)
            	at jadx.core.codegen.InsnGen.makeInvoke(InsnGen.java:794)
            	at jadx.core.codegen.InsnGen.makeInsnBody(InsnGen.java:401)
            	at jadx.core.codegen.InsnGen.addWrappedArg(InsnGen.java:143)
            	at jadx.core.codegen.InsnGen.addArg(InsnGen.java:119)
            	at jadx.core.codegen.InsnGen.addArg(InsnGen.java:106)
            	at jadx.core.codegen.InsnGen.generateMethodArguments(InsnGen.java:1075)
            	at jadx.core.codegen.InsnGen.makeInvoke(InsnGen.java:851)
            	at jadx.core.codegen.InsnGen.makeInsnBody(InsnGen.java:401)
            	at jadx.core.codegen.InsnGen.makeInsn(InsnGen.java:282)
            	... 19 more
            */
        /*
            this = this;
            r0 = r7
            if (r0 == 0) goto L16
            r0 = r8
            if (r0 == 0) goto Lc
            r0 = r9
            if (r0 != 0) goto L16
        Lc:
            java.lang.RuntimeException r0 = new java.lang.RuntimeException
            r1 = r0
            java.lang.String r2 = "ExternalTextureConverter: setSurfaceTexture dimensions cannot be zero"
            r1.<init>(r2)
            throw r0
        L16:
            r0 = r6
            com.google.mediapipe.components.ExternalTextureConverter$RenderThread r0 = r0.thread
            android.os.Handler r0 = r0.getHandler()
            r1 = r6
            r2 = r7
            r3 = r8
            r4 = r9
            void r1 = () -> { // java.lang.Runnable.run():void
                r1.lambda$setSurfaceTexture$1(r2, r3, r4);
            }
            boolean r0 = r0.post(r1)
            return
        */
        throw new UnsupportedOperationException("Method not decompiled: com.google.mediapipe.components.ExternalTextureConverter.setSurfaceTexture(android.graphics.SurfaceTexture, int, int):void");
    }

    public void setSurfaceTextureAndAttachToGLContext(SurfaceTexture texture, int width, int height) {
        if (texture != null && (width == 0 || height == 0)) {
            throw new RuntimeException("ExternalTextureConverter: setSurfaceTexture dimensions cannot be zero");
        }
        this.thread.getHandler().post(()
        /*  JADX ERROR: Method code generation error
            jadx.core.utils.exceptions.CodegenException: Error generate insn: 0x0026: INVOKE  
              (wrap: android.os.Handler : 0x001a: INVOKE  (r0v3 android.os.Handler A[REMOVE]) = 
              (wrap: com.google.mediapipe.components.ExternalTextureConverter$RenderThread : 0x0017: IGET  (r0v2 com.google.mediapipe.components.ExternalTextureConverter$RenderThread A[REMOVE]) = 
              (r6v0 'this' com.google.mediapipe.components.ExternalTextureConverter A[D('this' com.google.mediapipe.components.ExternalTextureConverter), IMMUTABLE_TYPE, THIS])
             com.google.mediapipe.components.ExternalTextureConverter.thread com.google.mediapipe.components.ExternalTextureConverter$RenderThread)
             type: VIRTUAL call: com.google.mediapipe.components.ExternalTextureConverter.RenderThread.getHandler():android.os.Handler)
              (wrap: java.lang.Runnable : 0x0021: INVOKE_CUSTOM (r1v1 java.lang.Runnable A[REMOVE]) = 
              (r6v0 'this' com.google.mediapipe.components.ExternalTextureConverter A[D('this' com.google.mediapipe.components.ExternalTextureConverter), DONT_INLINE, IMMUTABLE_TYPE, THIS])
              (r7v0 'texture' android.graphics.SurfaceTexture A[D('texture' android.graphics.SurfaceTexture), DONT_INLINE])
              (r8v0 'width' int A[D('width' int), DONT_INLINE])
              (r9v0 'height' int A[D('height' int), DONT_INLINE])
            
             handle type: INVOKE_DIRECT
             lambda: java.lang.Runnable.run():void
             call insn: ?: INVOKE  
              (r1 I:com.google.mediapipe.components.ExternalTextureConverter)
              (r2 I:android.graphics.SurfaceTexture)
              (r3 I:int)
              (r4 I:int)
             type: DIRECT call: com.google.mediapipe.components.ExternalTextureConverter.lambda$setSurfaceTextureAndAttachToGLContext$2(android.graphics.SurfaceTexture, int, int):void)
             type: VIRTUAL call: android.os.Handler.post(java.lang.Runnable):boolean in method: com.google.mediapipe.components.ExternalTextureConverter.setSurfaceTextureAndAttachToGLContext(android.graphics.SurfaceTexture, int, int):void, file: base.apk:classes.jar:com/google/mediapipe/components/ExternalTextureConverter.class
            	at jadx.core.codegen.InsnGen.makeInsn(InsnGen.java:289)
            	at jadx.core.codegen.InsnGen.makeInsn(InsnGen.java:252)
            	at jadx.core.codegen.RegionGen.makeSimpleBlock(RegionGen.java:91)
            	at jadx.core.dex.nodes.IBlock.generate(IBlock.java:15)
            	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
            	at jadx.core.dex.regions.Region.generate(Region.java:35)
            	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
            	at jadx.core.dex.regions.Region.generate(Region.java:35)
            	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
            	at jadx.core.dex.regions.Region.generate(Region.java:35)
            	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
            	at jadx.core.codegen.MethodGen.addRegionInsns(MethodGen.java:296)
            	at jadx.core.codegen.MethodGen.addInstructions(MethodGen.java:275)
            	at jadx.core.codegen.ClassGen.addMethodCode(ClassGen.java:377)
            	at jadx.core.codegen.ClassGen.addMethod(ClassGen.java:306)
            	at jadx.core.codegen.ClassGen.lambda$addInnerClsAndMethods$2(ClassGen.java:272)
            	at java.util.stream.ForEachOps$ForEachOp$OfRef.accept(ForEachOps.java:184)
            	at java.util.ArrayList.forEach(ArrayList.java:1259)
            	at java.util.stream.SortedOps$RefSortingSink.end(SortedOps.java:390)
            	at java.util.stream.Sink$ChainedReference.end(Sink.java:258)
            Caused by: java.lang.IndexOutOfBoundsException: Index: 3, Size: 3
            	at java.util.ArrayList.rangeCheck(ArrayList.java:659)
            	at java.util.ArrayList.get(ArrayList.java:435)
            	at jadx.core.codegen.InsnGen.makeInlinedLambdaMethod(InsnGen.java:998)
            	at jadx.core.codegen.InsnGen.makeInvokeLambda(InsnGen.java:903)
            	at jadx.core.codegen.InsnGen.makeInvoke(InsnGen.java:794)
            	at jadx.core.codegen.InsnGen.makeInsnBody(InsnGen.java:401)
            	at jadx.core.codegen.InsnGen.addWrappedArg(InsnGen.java:143)
            	at jadx.core.codegen.InsnGen.addArg(InsnGen.java:119)
            	at jadx.core.codegen.InsnGen.addArg(InsnGen.java:106)
            	at jadx.core.codegen.InsnGen.generateMethodArguments(InsnGen.java:1075)
            	at jadx.core.codegen.InsnGen.makeInvoke(InsnGen.java:851)
            	at jadx.core.codegen.InsnGen.makeInsnBody(InsnGen.java:401)
            	at jadx.core.codegen.InsnGen.makeInsn(InsnGen.java:282)
            	... 19 more
            */
        /*
            this = this;
            r0 = r7
            if (r0 == 0) goto L16
            r0 = r8
            if (r0 == 0) goto Lc
            r0 = r9
            if (r0 != 0) goto L16
        Lc:
            java.lang.RuntimeException r0 = new java.lang.RuntimeException
            r1 = r0
            java.lang.String r2 = "ExternalTextureConverter: setSurfaceTexture dimensions cannot be zero"
            r1.<init>(r2)
            throw r0
        L16:
            r0 = r6
            com.google.mediapipe.components.ExternalTextureConverter$RenderThread r0 = r0.thread
            android.os.Handler r0 = r0.getHandler()
            r1 = r6
            r2 = r7
            r3 = r8
            r4 = r9
            void r1 = () -> { // java.lang.Runnable.run():void
                r1.lambda$setSurfaceTextureAndAttachToGLContext$2(r2, r3, r4);
            }
            boolean r0 = r0.post(r1)
            return
        */
        throw new UnsupportedOperationException("Method not decompiled: com.google.mediapipe.components.ExternalTextureConverter.setSurfaceTextureAndAttachToGLContext(android.graphics.SurfaceTexture, int, int):void");
    }

    @Override // com.google.mediapipe.components.TextureFrameProducer
    public void setConsumer(TextureFrameConsumer next) {
        this.thread.setConsumer(next);
    }

    public void addConsumer(TextureFrameConsumer consumer) {
        this.thread.addConsumer(consumer);
    }

    public void removeConsumer(TextureFrameConsumer consumer) {
        this.thread.removeConsumer(consumer);
    }

    public void close() {
        if (this.thread == null) {
            return;
        }
        this.thread.quitSafely();
        try {
            this.thread.join();
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            Log.e("ExternalTextureConv", "thread was unexpectedly interrupted: " + ie.getMessage());
            throw new RuntimeException(ie);
        }
    }

    protected RenderThread makeRenderThread(EGLContext parentContext, int numBuffers) {
        return new RenderThread(parentContext, numBuffers);
    }

    /* JADX INFO: Access modifiers changed from: protected */
    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/components/ExternalTextureConverter$RenderThread.class */
    public static class RenderThread extends GlThread implements SurfaceTexture.OnFrameAvailableListener {
        private static final long NANOS_PER_MICRO = 1000;
        private volatile SurfaceTexture surfaceTexture;
        private final List<TextureFrameConsumer> consumers;
        private final Queue<PoolTextureFrame> framesAvailable;
        private int framesInUse;
        private final int framesToKeep;
        private ExternalTextureRenderer renderer;
        private long nextFrameTimestampOffset;
        private long timestampOffsetNanos;
        private long previousTimestamp;
        private boolean previousTimestampValid;
        protected int destinationWidth;
        protected int destinationHeight;

        /* JADX INFO: Access modifiers changed from: private */
        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/components/ExternalTextureConverter$RenderThread$PoolTextureFrame.class */
        public class PoolTextureFrame extends AppTextureFrame {
            public PoolTextureFrame(int textureName, int width, int height) {
                super(textureName, width, height);
            }

            @Override // com.google.mediapipe.framework.AppTextureFrame, com.google.mediapipe.framework.TextureFrame, com.google.mediapipe.framework.TextureReleaseCallback
            public void release(GlSyncToken syncToken) {
                super.release(syncToken);
                RenderThread.this.poolFrameReleased(this);
            }

            @Override // com.google.mediapipe.framework.AppTextureFrame, com.google.mediapipe.framework.TextureFrame
            public void release() {
                super.release();
                RenderThread.this.poolFrameReleased(this);
            }
        }

        public RenderThread(EGLContext parentContext, int numBuffers) {
            super(parentContext);
            this.surfaceTexture = null;
            this.framesAvailable = new ArrayDeque();
            this.framesInUse = 0;
            this.renderer = null;
            this.nextFrameTimestampOffset = 0L;
            this.timestampOffsetNanos = 0L;
            this.previousTimestamp = 0L;
            this.previousTimestampValid = false;
            this.destinationWidth = 0;
            this.destinationHeight = 0;
            this.framesToKeep = numBuffers;
            this.renderer = new ExternalTextureRenderer();
            this.consumers = new ArrayList();
        }

        public void setFlipY(boolean flip) {
            this.renderer.setFlipY(flip);
        }

        public void setRotation(int rotation) {
            this.renderer.setRotation(rotation);
        }

        public void setSurfaceTexture(SurfaceTexture texture, int width, int height) {
            if (this.surfaceTexture != null) {
                this.surfaceTexture.setOnFrameAvailableListener(null);
            }
            this.surfaceTexture = texture;
            if (this.surfaceTexture != null) {
                this.surfaceTexture.setOnFrameAvailableListener(this);
            }
            this.destinationWidth = width;
            this.destinationHeight = height;
        }

        public void setSurfaceTextureAndAttachToGLContext(SurfaceTexture texture, int width, int height) {
            setSurfaceTexture(texture, width, height);
            int[] textures = new int[1];
            GLES20.glGenTextures(1, textures, 0);
            this.surfaceTexture.attachToGLContext(textures[0]);
        }

        public void setConsumer(TextureFrameConsumer consumer) {
            synchronized (this.consumers) {
                this.consumers.clear();
                this.consumers.add(consumer);
            }
        }

        public void addConsumer(TextureFrameConsumer consumer) {
            synchronized (this.consumers) {
                this.consumers.add(consumer);
            }
        }

        public void removeConsumer(TextureFrameConsumer consumer) {
            synchronized (this.consumers) {
                this.consumers.remove(consumer);
            }
        }

        @Override // android.graphics.SurfaceTexture.OnFrameAvailableListener
        public void onFrameAvailable(SurfaceTexture surfaceTexture) {
            this.handler.post(()
            /*  JADX ERROR: Method code generation error
                jadx.core.utils.exceptions.CodegenException: Error generate insn: 0x000b: INVOKE  
                  (wrap: android.os.Handler : 0x0001: IGET  (r0v1 android.os.Handler A[REMOVE]) = 
                  (r4v0 'this' com.google.mediapipe.components.ExternalTextureConverter$RenderThread A[D('this' com.google.mediapipe.components.ExternalTextureConverter$RenderThread), IMMUTABLE_TYPE, THIS])
                 com.google.mediapipe.components.ExternalTextureConverter.RenderThread.handler android.os.Handler)
                  (wrap: java.lang.Runnable : 0x0006: INVOKE_CUSTOM (r1v1 java.lang.Runnable A[REMOVE]) = 
                  (r4v0 'this' com.google.mediapipe.components.ExternalTextureConverter$RenderThread A[D('this' com.google.mediapipe.components.ExternalTextureConverter$RenderThread), DONT_INLINE, IMMUTABLE_TYPE, THIS])
                  (r5v0 'surfaceTexture' android.graphics.SurfaceTexture A[D('surfaceTexture' android.graphics.SurfaceTexture), DONT_INLINE])
                
                 handle type: INVOKE_DIRECT
                 lambda: java.lang.Runnable.run():void
                 call insn: ?: INVOKE  (r1 I:com.google.mediapipe.components.ExternalTextureConverter$RenderThread), (r2 I:android.graphics.SurfaceTexture) type: DIRECT call: com.google.mediapipe.components.ExternalTextureConverter.RenderThread.lambda$onFrameAvailable$0(android.graphics.SurfaceTexture):void)
                 type: VIRTUAL call: android.os.Handler.post(java.lang.Runnable):boolean in method: com.google.mediapipe.components.ExternalTextureConverter.RenderThread.onFrameAvailable(android.graphics.SurfaceTexture):void, file: base.apk:classes.jar:com/google/mediapipe/components/ExternalTextureConverter$RenderThread.class
                	at jadx.core.codegen.InsnGen.makeInsn(InsnGen.java:289)
                	at jadx.core.codegen.InsnGen.makeInsn(InsnGen.java:252)
                	at jadx.core.codegen.RegionGen.makeSimpleBlock(RegionGen.java:91)
                	at jadx.core.dex.nodes.IBlock.generate(IBlock.java:15)
                	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
                	at jadx.core.dex.regions.Region.generate(Region.java:35)
                	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
                	at jadx.core.codegen.MethodGen.addRegionInsns(MethodGen.java:296)
                	at jadx.core.codegen.MethodGen.addInstructions(MethodGen.java:275)
                	at jadx.core.codegen.ClassGen.addMethodCode(ClassGen.java:377)
                	at jadx.core.codegen.ClassGen.addMethod(ClassGen.java:306)
                	at jadx.core.codegen.ClassGen.lambda$addInnerClsAndMethods$2(ClassGen.java:272)
                	at java.util.stream.ForEachOps$ForEachOp$OfRef.accept(ForEachOps.java:184)
                	at java.util.ArrayList.forEach(ArrayList.java:1259)
                	at java.util.stream.SortedOps$RefSortingSink.end(SortedOps.java:390)
                	at java.util.stream.Sink$ChainedReference.end(Sink.java:258)
                Caused by: java.lang.IndexOutOfBoundsException: Index: 1, Size: 1
                	at java.util.ArrayList.rangeCheck(ArrayList.java:659)
                	at java.util.ArrayList.get(ArrayList.java:435)
                	at jadx.core.codegen.InsnGen.makeInlinedLambdaMethod(InsnGen.java:998)
                	at jadx.core.codegen.InsnGen.makeInvokeLambda(InsnGen.java:903)
                	at jadx.core.codegen.InsnGen.makeInvoke(InsnGen.java:794)
                	at jadx.core.codegen.InsnGen.makeInsnBody(InsnGen.java:401)
                	at jadx.core.codegen.InsnGen.addWrappedArg(InsnGen.java:143)
                	at jadx.core.codegen.InsnGen.addArg(InsnGen.java:119)
                	at jadx.core.codegen.InsnGen.addArg(InsnGen.java:106)
                	at jadx.core.codegen.InsnGen.generateMethodArguments(InsnGen.java:1075)
                	at jadx.core.codegen.InsnGen.makeInvoke(InsnGen.java:851)
                	at jadx.core.codegen.InsnGen.makeInsnBody(InsnGen.java:401)
                	at jadx.core.codegen.InsnGen.makeInsn(InsnGen.java:282)
                	... 15 more
                */
            /*
                this = this;
                r0 = r4
                android.os.Handler r0 = r0.handler
                r1 = r4
                r2 = r5
                void r1 = () -> { // java.lang.Runnable.run():void
                    r1.lambda$onFrameAvailable$0(r2);
                }
                boolean r0 = r0.post(r1)
                return
            */
            throw new UnsupportedOperationException("Method not decompiled: com.google.mediapipe.components.ExternalTextureConverter.RenderThread.onFrameAvailable(android.graphics.SurfaceTexture):void");
        }

        @Override // com.google.mediapipe.glutil.GlThread
        public void prepareGl() {
            super.prepareGl();
            GLES20.glClearColor(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f);
            this.renderer.setup();
        }

        @Override // com.google.mediapipe.glutil.GlThread
        public void releaseGl() {
            setSurfaceTexture(null, 0, 0);
            while (!this.framesAvailable.isEmpty()) {
                teardownFrame(this.framesAvailable.remove());
            }
            this.renderer.release();
            super.releaseGl();
        }

        public void setTimestampOffsetNanos(long offsetInNanos) {
            this.timestampOffsetNanos = offsetInNanos;
        }

        /* JADX INFO: Access modifiers changed from: protected */
        public void renderNext(SurfaceTexture fromTexture) {
            if (fromTexture != this.surfaceTexture) {
                return;
            }
            synchronized (this.consumers) {
                boolean frameUpdated = false;
                for (TextureFrameConsumer consumer : this.consumers) {
                    AppTextureFrame outputFrame = nextOutputFrame();
                    updateOutputFrame(outputFrame);
                    frameUpdated = true;
                    if (consumer != null) {
                        if (Log.isLoggable("ExternalTextureConv", 2)) {
                            Log.v("ExternalTextureConv", String.format("Locking tex: %d width: %d height: %d", Integer.valueOf(outputFrame.getTextureName()), Integer.valueOf(outputFrame.getWidth()), Integer.valueOf(outputFrame.getHeight())));
                        }
                        outputFrame.setInUse();
                        consumer.onNewFrame(outputFrame);
                    }
                }
                if (!frameUpdated) {
                    updateOutputFrame(nextOutputFrame());
                }
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public static void teardownFrame(AppTextureFrame frame) {
            GLES20.glDeleteTextures(1, new int[]{frame.getTextureName()}, 0);
        }

        private PoolTextureFrame createFrame() {
            int destinationTextureId = ShaderUtil.createRgbaTexture(this.destinationWidth, this.destinationHeight);
            Log.d("ExternalTextureConv", String.format("Created output texture: %d width: %d height: %d", Integer.valueOf(destinationTextureId), Integer.valueOf(this.destinationWidth), Integer.valueOf(this.destinationHeight)));
            bindFramebuffer(destinationTextureId, this.destinationWidth, this.destinationHeight);
            return new PoolTextureFrame(destinationTextureId, this.destinationWidth, this.destinationHeight);
        }

        private AppTextureFrame nextOutputFrame() {
            PoolTextureFrame outputFrame;
            synchronized (this) {
                outputFrame = this.framesAvailable.poll();
                this.framesInUse++;
            }
            if (outputFrame == null) {
                outputFrame = createFrame();
            } else if (outputFrame.getWidth() != this.destinationWidth || outputFrame.getHeight() != this.destinationHeight) {
                waitUntilReleased(outputFrame);
                teardownFrame(outputFrame);
                outputFrame = createFrame();
            } else {
                waitUntilReleased(outputFrame);
            }
            return outputFrame;
        }

        protected synchronized void poolFrameReleased(PoolTextureFrame frame) {
            this.framesAvailable.offer(frame);
            this.framesInUse--;
            int keep = Math.max(this.framesToKeep - this.framesInUse, 0);
            while (this.framesAvailable.size() > keep) {
                PoolTextureFrame textureFrameToRemove = this.framesAvailable.remove();
                this.handler.post(() -> {
                    teardownFrame(textureFrameToRemove);
                });
            }
        }

        private void updateOutputFrame(AppTextureFrame outputFrame) {
            bindFramebuffer(outputFrame.getTextureName(), this.destinationWidth, this.destinationHeight);
            this.renderer.render(this.surfaceTexture);
            long textureTimestamp = (this.surfaceTexture.getTimestamp() + this.timestampOffsetNanos) / 1000;
            if (this.previousTimestampValid && textureTimestamp + this.nextFrameTimestampOffset <= this.previousTimestamp) {
                this.nextFrameTimestampOffset = (this.previousTimestamp + 1) - textureTimestamp;
            }
            outputFrame.setTimestamp(textureTimestamp + this.nextFrameTimestampOffset);
            this.previousTimestamp = outputFrame.getTimestamp();
            this.previousTimestampValid = true;
        }

        private void waitUntilReleased(AppTextureFrame frame) {
            try {
                if (Log.isLoggable("ExternalTextureConv", 2)) {
                    Log.v("ExternalTextureConv", String.format("Waiting for tex: %d width: %d height: %d timestamp: %d", Integer.valueOf(frame.getTextureName()), Integer.valueOf(frame.getWidth()), Integer.valueOf(frame.getHeight()), Long.valueOf(frame.getTimestamp())));
                }
                frame.waitUntilReleased();
                if (Log.isLoggable("ExternalTextureConv", 2)) {
                    Log.v("ExternalTextureConv", String.format("Finished waiting for tex: %d width: %d height: %d timestamp: %d", Integer.valueOf(frame.getTextureName()), Integer.valueOf(frame.getWidth()), Integer.valueOf(frame.getHeight()), Long.valueOf(frame.getTimestamp())));
                }
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                Log.e("ExternalTextureConv", "thread was unexpectedly interrupted: " + ie.getMessage());
                throw new RuntimeException(ie);
            }
        }
    }
}