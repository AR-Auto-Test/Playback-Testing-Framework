package androidx.media;

import android.util.SparseIntArray;
import b.b0.c;

/* loaded from: classes.dex */
public class AudioAttributesCompat implements c {

    /* renamed from: a  reason: collision with root package name */
    public static final SparseIntArray f334a;

    /* renamed from: b  reason: collision with root package name */
    public AudioAttributesImpl f335b;

    static {
        SparseIntArray sparseIntArray = new SparseIntArray();
        f334a = sparseIntArray;
        sparseIntArray.put(5, 1);
        sparseIntArray.put(6, 2);
        sparseIntArray.put(7, 2);
        sparseIntArray.put(8, 1);
        sparseIntArray.put(9, 1);
        sparseIntArray.put(10, 1);
    }

    public boolean equals(Object obj) {
        if (obj instanceof AudioAttributesCompat) {
            AudioAttributesCompat audioAttributesCompat = (AudioAttributesCompat) obj;
            AudioAttributesImpl audioAttributesImpl = this.f335b;
            if (audioAttributesImpl == null) {
                return audioAttributesCompat.f335b == null;
            }
            return audioAttributesImpl.equals(audioAttributesCompat.f335b);
        }
        return false;
    }

    public int hashCode() {
        return this.f335b.hashCode();
    }

    public String toString() {
        return this.f335b.toString();
    }
}