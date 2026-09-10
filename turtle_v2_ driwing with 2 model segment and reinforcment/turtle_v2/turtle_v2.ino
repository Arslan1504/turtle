#include <Dynamixel2Arduino.h>

Dynamixel2Arduino dxl(Serial3, 84);

String tempCommand = "";
String latestValidCommand = "";
unsigned long previousMillis = 0;

// NEW: Timer to track the last time we heard from Python
unsigned long lastCommandTime = 0; 

void setup() {
  Serial.begin(115200); 
  
  pinMode(BDPIN_DXL_PWR_EN, 1);
  digitalWrite(BDPIN_DXL_PWR_EN, 1);
  delay(500);

  dxl.begin(1000000);
  dxl.setPortProtocolVersion(2.0);

  dxl.torqueOff(1);
  dxl.torqueOff(2);
  dxl.setOperatingMode(1, OP_VELOCITY); 
  dxl.setOperatingMode(2, OP_VELOCITY);
  dxl.torqueOn(1);
  dxl.torqueOn(2);
}

void loop() {
  // 1. Drain the buffer
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      latestValidCommand = tempCommand;
      tempCommand = "";
      
      // We received a valid command! Reset the failsafe timer.
      lastCommandTime = millis(); 
    } else if (c != '\r') {
      tempCommand += c;
    }
  }

  // 2. Apply the newest command
  if (latestValidCommand.length() > 0) {
    int commaIndex = latestValidCommand.indexOf(',');
    if (commaIndex > 0) {
      float leftSpeed = latestValidCommand.substring(0, commaIndex).toFloat();
      float rightSpeed = latestValidCommand.substring(commaIndex + 1).toFloat();
      
      dxl.setGoalVelocity(1, leftSpeed, UNIT_RPM);
      dxl.setGoalVelocity(2, rightSpeed, UNIT_RPM);
    }
    latestValidCommand = ""; 
  }

  // ---------------------------------------------------------
  // 3. FAILSAFE WATCHDOG: Stop motors if Python disconnected
  // ---------------------------------------------------------
  // If 500ms (0.5 seconds) pass without a new command, hit the brakes!
  if (millis() - lastCommandTime > 500) {
    dxl.setGoalVelocity(1, 0, UNIT_RPM);
    dxl.setGoalVelocity(2, 0, UNIT_RPM);
  }

  // 4. Send the Sync signal back to Python
  if (millis() - previousMillis >= 50) {
    previousMillis = millis();
    Serial.print("E,0,0\n"); 
  }
}