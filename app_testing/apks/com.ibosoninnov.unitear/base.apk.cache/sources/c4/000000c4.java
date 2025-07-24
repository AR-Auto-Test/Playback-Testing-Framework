package androidx.media;

import android.media.AudioAttributes;
import b.b0.a;
import java.util.Objects;

/* loaded from: classes.dex */
public final class AudioAttributesImplApi21Parcelizer {
    public static AudioAttributesImplApi21 read(a aVar) {
        AudioAttributesImplApi21 audioAttributesImplApi21 = new AudioAttributesImplApi21();
        audioAttributesImplApi21.f336a = (AudioAttributes) aVar.m(audioAttributesImplApi21.f336a, 1);
        audioAttributesImplApi21.f337b = aVar.k(audioAttributesImplApi21.f337b, 2);
        return audioAttributesImplApi21;
    }

    public static void write(AudioAttributesImplApi21 audioAttributesImplApi21, a aVar) {
        Objects.requireNonNull(aVar);
        AudioAttributes audioAttributes = audioAttributesImplApi21.f336a;
        aVar.p(1);
        aVar.u(audioAttributes);
        int i = audioAttributesImplApi21.f337b;
        aVar.p(2);
        aVar.t(i);
    }
}