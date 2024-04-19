#include <lmic.h>
#include <hal/hal.h>
#include <SPI.h>


#define TX_INTERVAL 1000

static const PROGMEM u1_t NWKSKEY[16] = {0x47,0xBC,0xD1,0xFE,0xF6,0x5B,0x8C,0xBB,0xD4,0x38,0x78,0x34,0x10,0x43,0xDE,0x4A};

static const u1_t PROGMEM APPSKEY[16] = {0x9D,0xB0,0xC4,0xB1,0xF5,0x92,0x63,0xA0,0x1A,0x1A,0x19,0xF9,0x1E,0x75,0xA5,0x09};

// static const u4_t DEVADDR = 0x01011688; // <-- Change this address for every node!
static const u4_t DEVADDR = 0x00057D03; // <-- Change this address for every node! 

osjob_t txjob;
osjob_t rxjob;
osjob_t timeoutjob;
static osjob_t sendjob;

const lmic_pinmap lmic_pins = {
    .nss = 10,
    .rxtx = LMIC_UNUSED_PIN,
    .rst = 9, 
    .dio = {2, 5, 6},
};

char mydata[40];

void os_getArtEui (u1_t* buf) { }
void os_getDevEui (u1_t* buf) { }
void os_getDevKey (u1_t* buf) { }

void onEvent (ev_t ev) {
}

// Transmit the given string and call the given function afterwards
void tx(const char *str, osjobcb_t func) {
  os_radio(RADIO_RST); // Stop RX first
  delay(1); // Wait a bit, without this os_radio below asserts, apparently because the state hasn't changed yet
  LMIC.dataLen = 0;
  while (*str)
  LMIC.frame[LMIC.dataLen++] = *str++;
  LMIC.osjob.func = func;
  os_radio(RADIO_TX);
}

void set_radio_param()
{
  #ifdef VCC_ENABLE
    // For Pinoccio Scout boards
    pinMode(VCC_ENABLE, OUTPUT);
    digitalWrite(VCC_ENABLE, HIGH);
    delay(1000);
    #endif
    // LMIC init
    os_init();
    // Reset the MAC state. Session and pending data transfers will be discarded.
    LMIC_reset();
    // Set static session parameters. Instead of dynamically establishing a session
    // by joining the network, precomputed session parameters are be provided.
    #ifdef PROGMEM
    // On AVR, these values are stored in flash and only copied to RAM
    // once. Copy them to a temporary buffer here, LMIC_setSession will
    // copy them into a buffer of its own again.
    uint8_t appskey[sizeof(APPSKEY)];
    uint8_t nwkskey[sizeof(NWKSKEY)];
    memcpy_P(appskey, APPSKEY, sizeof(APPSKEY));
    memcpy_P(nwkskey, NWKSKEY, sizeof(NWKSKEY));
    LMIC_setSession (0x1, DEVADDR, nwkskey, appskey);
    #else
    // If not running an AVR with PROGMEM, just use the arrays directly
    LMIC_setSession (0x1, DEVADDR, NWKSKEY, APPSKEY);
    #endif

      /* TTN uses SF9 for its RX2 window. */
  // LMIC.dn2Dr = DR_SF10;

  // // Set up these settings once, and use them for both TX and RX
  // LMIC.freq = 904300000; //904300000 # 917200000
  // LMIC.txpow = 14; //#
  // LMIC.datarate = DR_SF10;
  // LMIC.errcr = CR_4_5;
  // LMIC.rps = setCr(updr2rps(LMIC.datarate), LMIC.errcr);
    // Start job
  // do_send(&sendjob);
}

// Enable rx mode and call func when a packet is received
void rx(osjobcb_t func) {
  LMIC.osjob.func = func;
  LMIC.rxtime = os_getTime(); // RX _now_

  os_radio(RADIO_RXON);
}

static void rxtimeout_func(osjob_t *job) {
  digitalWrite(LED_BUILTIN, LOW); // off
}

static void rx_func (osjob_t* job) {
  
  // Blink once to confirm reception and then keep the led on
  digitalWrite(LED_BUILTIN, LOW); // off
  delay(10);
  digitalWrite(LED_BUILTIN, HIGH); // on

  os_setTimedCallback(&txjob, os_getTime(), tx_func); //+ ms2osticks(TX_INTERVAL/2)

  rx(rx_func);
}

static void txdone_func (osjob_t* job) {
  rx(rx_func);
}

// log text to USART and toggle LED
static void tx_func (osjob_t* job) {
  
  tx(mydata, txdone_func);
  
  byte c = 0;
  // for(c = sizeof(mydata) - 2; c >= 0; c--){
  //   if(mydata[c] == '9') mydata[c] = '0';
  //   else {
  //     mydata[c]++;
  //     break;
  //   }
  // }

  for(c=0;c<sizeof(mydata);c++) {
    Serial.print(mydata[c]);
  }
  
  Serial.println();

  //delay(TX_INTERVAL)
  os_setTimedCallback(job, os_getTime()+ ms2osticks(TX_INTERVAL), tx_func);
}

void setup() {
  set_radio_param();
  Serial.begin(9600); //sets the baud rate
  pinMode(7, OUTPUT);
  pinMode(4, OUTPUT);
  
  #ifdef PROGMEM
  uint8_t appskey[sizeof(APPSKEY)];
  uint8_t nwkskey[sizeof(NWKSKEY)];
  memcpy_P(appskey, APPSKEY, sizeof(APPSKEY));
  memcpy_P(nwkskey, NWKSKEY, sizeof(NWKSKEY));
  #endif
  
  #ifdef VCC_ENABLE

  pinMode(VCC_ENABLE, OUTPUT);
  digitalWrite(VCC_ENABLE, HIGH);
  delay(1000);
  #endif
  pinMode(LED_BUILTIN, OUTPUT);

  os_init();

  // Set up these settings once, and use them for both TX and RX
  int randomIndex = random(0, 6);
  int freqList[6] = {904100000, 904300000, 904500000, 904700000, 904900000, 905100000};

  // LMIC.freq = freqList[randomIndex]; //904300000 # 917200000
  LMIC.freq = 904500000;
  LMIC.txpow = 14; //#
  LMIC.datarate = DR_SF10;
  LMIC.errcr = CR_4_5;
  LMIC.rps = setCr(updr2rps(LMIC.datarate), LMIC.errcr);
  
  Serial.println("Started");
  Serial.flush();
  
  os_setCallback(&txjob, tx_func);
}

void loop()
{
  digitalWrite(7, HIGH);
  digitalWrite(4, HIGH);
  float v_oxygen_raw1;
  float v_swt_raw1;
  float v_oxygen_raw2;
  float v_swt_raw2;
  float v_swt1;
  float v_swt2;
//  float oxy = 0.0;
  float v_oxygen1 = 0.0;
  float v_oxygen2 = 0.0;
  pinMode(A0,INPUT_PULLUP);
  pinMode(A1,INPUT_PULLUP);
  pinMode(A2,INPUT_PULLUP);
  pinMode(A5,INPUT_PULLUP);

  v_swt_raw1 = analogRead(A0); 
  v_oxygen_raw1 = analogRead(A1); 
  v_swt_raw2 = analogRead(A5); 
  v_oxygen_raw2 = analogRead(A2); 

  /*reads the analog value from the Oxygen gas sensor and this value can
  be anywhere from 0 to 1023 (10 bit ADC used in Arduino Uno)*/ 
  if(v_oxygen_raw1>900) //To make sure that reading is zero when the circuit is open
  {
      v_oxygen_raw1=0;
  }
  v_oxygen1 = (v_oxygen_raw1*3000)/992; 

  if(v_oxygen_raw2>900) //To make sure that reading is zero when the circuit is open
  {
      v_oxygen_raw2=0;
  }
  v_oxygen2 = (v_oxygen_raw2*3000)/992; 

  /*The above line calculates the output of the sensor in mV using the
  fact that a HIGH reading in an input pin of an Arduino Uno board is
  equivalent to a voltage of 3V or higher. When I displayed just 'v1',
  the lowest of the high values I got was 992. So I equated it to 3V
  (3000mV) and used the unitary method to find out what a non-HIGH v1 
  value would be in terms of mV.*/
  v_swt1 = (v_swt_raw1*3000)/992; 
  v_swt2 = (v_swt_raw2*3000)/992; 

  Serial.print("O2:");
  Serial.print(v_oxygen1);
  Serial.print(",");
  Serial.print(v_oxygen2);

  Serial.print(", SWT:");
  Serial.print(v_swt1);
  Serial.print(",");
  Serial.println(v_swt2);

  char str_oxygen1[10];  // Buffer for the converted float
  char str_oxygen2[10];  // Buffer for the converted float
  char str_swt1[10];     // Buffer for the converted float
  char str_swt2[10];     // Buffer for the converted float

  // Convert the floats to strings
  dtostrf(v_oxygen1, 6, 2, str_oxygen1); // 6 is the min field width; 2 is the number of digits after the decimal
  dtostrf(v_oxygen2, 6, 2, str_oxygen2);
  dtostrf(v_swt1, 6, 2, str_swt1);
  dtostrf(v_swt2, 6, 2, str_swt2);
  snprintf(mydata, sizeof(mydata), "E&F O2:%s,%s, SWT:%s,%s\n", str_oxygen1, str_oxygen2, str_swt1, str_swt2);

  // Format the string and store it in `mydata`
  // snprintf(mydata, sizeof(mydata), "3&4 O2:%.2f,%.2f, SWT:%.2f,%.2f\n", v_oxygen1, v_oxygen2, v_swt1, v_swt2);
  delay(80000);
  os_runloop_once();
}