package h.a.a;

import android.graphics.Bitmap;
import android.graphics.drawable.Drawable;
import android.os.Parcel;
import android.os.Parcelable;
import android.view.View;
import pl.droidsonroids.gif.GifInfoHandle;

/* compiled from: GifViewSavedState.java */
/* loaded from: classes2.dex */
public class f extends View.BaseSavedState {
    public static final Parcelable.Creator<f> CREATOR = new a();

    /* renamed from: b  reason: collision with root package name */
    public final long[][] f6244b;

    /* compiled from: GifViewSavedState.java */
    /* loaded from: classes2.dex */
    public static class a implements Parcelable.Creator<f> {
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.os.Parcelable.Creator
        public f createFromParcel(Parcel parcel) {
            return new f(parcel, (a) null);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
        @Override // android.os.Parcelable.Creator
        public f[] newArray(int i) {
            return new f[i];
        }
    }

    public f(Parcelable parcelable, Drawable... drawableArr) {
        super(parcelable);
        long[] savedState;
        this.f6244b = new long[drawableArr.length];
        for (int i = 0; i < drawableArr.length; i++) {
            Drawable drawable = drawableArr[i];
            if (drawable instanceof c) {
                long[][] jArr = this.f6244b;
                GifInfoHandle gifInfoHandle = ((c) drawable).f6232h;
                synchronized (gifInfoHandle) {
                    savedState = GifInfoHandle.getSavedState(gifInfoHandle.f6268b);
                }
                jArr[i] = savedState;
            } else {
                this.f6244b[i] = null;
            }
        }
    }

    public void a(Drawable drawable, int i) {
        int restoreSavedState;
        long[][] jArr = this.f6244b;
        if (jArr[i] == null || !(drawable instanceof c)) {
            return;
        }
        c cVar = (c) drawable;
        GifInfoHandle gifInfoHandle = cVar.f6232h;
        long[] jArr2 = jArr[i];
        Bitmap bitmap = cVar.f6231g;
        synchronized (gifInfoHandle) {
            restoreSavedState = GifInfoHandle.restoreSavedState(gifInfoHandle.f6268b, jArr2, bitmap);
        }
        cVar.a(restoreSavedState);
    }

    @Override // android.view.View.BaseSavedState, android.view.AbsSavedState, android.os.Parcelable
    public void writeToParcel(Parcel parcel, int i) {
        super.writeToParcel(parcel, i);
        parcel.writeInt(this.f6244b.length);
        for (long[] jArr : this.f6244b) {
            parcel.writeLongArray(jArr);
        }
    }

    public f(Parcel parcel, a aVar) {
        super(parcel);
        this.f6244b = new long[parcel.readInt()];
        int i = 0;
        while (true) {
            long[][] jArr = this.f6244b;
            if (i >= jArr.length) {
                return;
            }
            jArr[i] = parcel.createLongArray();
            i++;
        }
    }
}