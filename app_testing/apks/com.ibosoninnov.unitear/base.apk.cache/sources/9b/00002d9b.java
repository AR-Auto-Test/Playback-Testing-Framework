package h.a.a;

import android.graphics.Bitmap;
import android.os.SystemClock;
import java.util.concurrent.TimeUnit;
import pl.droidsonroids.gif.GifInfoHandle;

/* compiled from: RenderTask.java */
/* loaded from: classes2.dex */
public class j extends k {
    public j(c cVar) {
        super(cVar);
    }

    @Override // h.a.a.k
    public void a() {
        long renderFrame;
        int currentFrameIndex;
        int currentLoop;
        int loopCount;
        c cVar = this.f6252b;
        GifInfoHandle gifInfoHandle = cVar.f6232h;
        Bitmap bitmap = cVar.f6231g;
        synchronized (gifInfoHandle) {
            renderFrame = GifInfoHandle.renderFrame(gifInfoHandle.f6268b, bitmap);
        }
        if (renderFrame >= 0) {
            this.f6252b.f6228d = SystemClock.uptimeMillis() + renderFrame;
            if (this.f6252b.isVisible() && this.f6252b.f6227c) {
                c cVar2 = this.f6252b;
                if (!cVar2.m) {
                    cVar2.f6226b.remove(this);
                    c cVar3 = this.f6252b;
                    cVar3.q = cVar3.f6226b.schedule(this, renderFrame, TimeUnit.MILLISECONDS);
                }
            }
            if (!this.f6252b.i.isEmpty()) {
                GifInfoHandle gifInfoHandle2 = this.f6252b.f6232h;
                synchronized (gifInfoHandle2) {
                    currentFrameIndex = GifInfoHandle.getCurrentFrameIndex(gifInfoHandle2.f6268b);
                }
                if (currentFrameIndex == this.f6252b.f6232h.b() - 1) {
                    c cVar4 = this.f6252b;
                    h hVar = cVar4.n;
                    GifInfoHandle gifInfoHandle3 = cVar4.f6232h;
                    synchronized (gifInfoHandle3) {
                        currentLoop = GifInfoHandle.getCurrentLoop(gifInfoHandle3.f6268b);
                    }
                    if (currentLoop != 0) {
                        GifInfoHandle gifInfoHandle4 = cVar4.f6232h;
                        synchronized (gifInfoHandle4) {
                            loopCount = GifInfoHandle.getLoopCount(gifInfoHandle4.f6268b);
                        }
                        if (currentLoop >= loopCount) {
                            currentLoop--;
                        }
                    }
                    hVar.sendEmptyMessageAtTime(currentLoop, this.f6252b.f6228d);
                }
            }
        } else {
            c cVar5 = this.f6252b;
            cVar5.f6228d = Long.MIN_VALUE;
            cVar5.f6227c = false;
        }
        if (!this.f6252b.isVisible() || this.f6252b.n.hasMessages(-1)) {
            return;
        }
        this.f6252b.n.sendEmptyMessageAtTime(-1, 0L);
    }
}