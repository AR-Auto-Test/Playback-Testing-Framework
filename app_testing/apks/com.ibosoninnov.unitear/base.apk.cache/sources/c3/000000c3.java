package androidx.media;

import android.annotation.TargetApi;
import android.media.AudioAttributes;
import c.b.a.a.a;

@TargetApi(21)
/* loaded from: classes.dex */
public class AudioAttributesImplApi21 implements AudioAttributesImpl {

    /* renamed from: a  reason: collision with root package name */
    public AudioAttributes f336a;

    /* renamed from: b  reason: collision with root package name */
    public int f337b = -1;

    public boolean equals(Object obj) {
        if (obj instanceof AudioAttributesImplApi21) {
            return this.f336a.equals(((AudioAttributesImplApi21) obj).f336a);
        }
        return false;
    }

    public int hashCode() {
        return this.f336a.hashCode();
    }

    public String toString() {
        StringBuilder x = a.x("AudioAttributesCompat: audioattributes=");
        x.append(this.f336a);
        return x.toString();
    }
}