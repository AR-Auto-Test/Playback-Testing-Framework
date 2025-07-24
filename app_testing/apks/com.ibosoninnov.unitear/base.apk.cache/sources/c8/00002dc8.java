package org.opencv.android;

import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.Log;
import java.text.DecimalFormat;
import org.opencv.core.Core;

/* loaded from: classes2.dex */
public class FpsMeter {
    private static final DecimalFormat FPS_FORMAT = new DecimalFormat("0.00");
    private static final int STEP = 20;
    private static final String TAG = "FpsMeter";
    private int mFramesCounter;
    private double mFrequency;
    public Paint mPaint;
    private String mStrfps;
    private long mprevFrameTime;
    public boolean mIsInitialized = false;
    public int mWidth = 0;
    public int mHeight = 0;

    public void draw(Canvas canvas, float f2, float f3) {
        Log.d(TAG, this.mStrfps);
        canvas.drawText(this.mStrfps, f2, f3, this.mPaint);
    }

    public void init() {
        this.mFramesCounter = 0;
        this.mFrequency = Core.getTickFrequency();
        this.mprevFrameTime = Core.getTickCount();
        this.mStrfps = "";
        Paint paint = new Paint();
        this.mPaint = paint;
        paint.setColor(-16776961);
        this.mPaint.setTextSize(20.0f);
    }

    public void measure() {
        if (!this.mIsInitialized) {
            init();
            this.mIsInitialized = true;
            return;
        }
        int i = this.mFramesCounter + 1;
        this.mFramesCounter = i;
        if (i % 20 == 0) {
            long tickCount = Core.getTickCount();
            double d2 = (this.mFrequency * 20.0d) / (tickCount - this.mprevFrameTime);
            this.mprevFrameTime = tickCount;
            if (this.mWidth != 0 && this.mHeight != 0) {
                this.mStrfps = FPS_FORMAT.format(d2) + " FPS@" + Integer.valueOf(this.mWidth) + "x" + Integer.valueOf(this.mHeight);
            } else {
                this.mStrfps = FPS_FORMAT.format(d2) + " FPS";
            }
            Log.i(TAG, this.mStrfps);
        }
    }

    public void setResolution(int i, int i2) {
        this.mWidth = i;
        this.mHeight = i2;
    }
}